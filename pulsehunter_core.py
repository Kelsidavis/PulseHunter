import json
import os
import subprocess
from datetime import datetime

import astropy.units as u
import cv2
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from PySide6.QtWidgets import QMessageBox


def plate_solve_astap(filepath, astap_exe="astap"):
    try:
        result = subprocess.run(
            [astap_exe, "-f", filepath, "-solve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"ASTAP failed: {result.stderr.decode().strip()}")
        else:
            print(f"ASTAP solved: {filepath}")
    except Exception as e:
        print(f"ASTAP error: {e}")


def load_fits_stack(
    folder,
    plate_solve_missing=False,
    astap_exe="astap",
    master_bias=None,
    master_dark=None,
    master_flat=None,
    camera_mode="mono",
    filter_name=None,
):
    frames = []
    filenames = []
    wcs_objects = []

    for file in sorted(os.listdir(folder)):
        if file.endswith(".fits"):
            path = os.path.join(folder, file)

            try:
                hdr = fits.getheader(path)
                has_wcs = "CRVAL1" in hdr and "CRVAL2" in hdr
            except Exception:
                has_wcs = False

            if plate_solve_missing and not has_wcs:
                plate_solve_astap(path, astap_exe)

            hdul = fits.open(path)
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            hdul.close()

            if master_bias is not None:
                data -= master_bias
            if master_dark is not None:
                data -= master_dark
            if master_flat is not None and np.all(master_flat != 0):
                data /= master_flat

            if camera_mode == "osc":
                pass  # Reserved for future OSC handling

            frames.append(data)
            filenames.append(file)

            try:
                wcs = WCS(header)
                if not wcs.has_celestial:
                    wcs = None
            except Exception:
                wcs = None

            wcs_objects.append(wcs)

    return np.array(frames), filenames, wcs_objects


def detect_transients(
    frames,
    filenames,
    wcs_objects,
    output_dir="detections",
    z_thresh=6.0,
    cutout_size=50,
    edge_margin=20,
    detect_dimming=False,
):
    if len(frames) == 0:
        print("No frames loaded.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    mean_image = np.mean(frames, axis=0)
    std_image = np.std(frames, axis=0)

    detections = []

    for i, frame in enumerate(frames):
        z = (frame - mean_image) / (std_image + 1e-5)
        y, x = np.unravel_index(np.argmax(np.abs(z)), z.shape)
        z_value = z[y, x]

        if abs(z_value) > z_thresh:
            if (
                x < edge_margin
                or y < edge_margin
                or x > frame.shape[1] - edge_margin
                or y > frame.shape[0] - edge_margin
            ):
                continue

            cutout = frame[
                max(0, y - cutout_size // 2) : y + cutout_size // 2,
                max(0, x - cutout_size // 2) : x + cutout_size // 2,
            ]
            cutout_norm = (
                255
                * (cutout - np.min(cutout))
                / (np.max(cutout) - np.min(cutout) + 1e-5)
            )
            cutout_img = cutout_norm.astype(np.uint8)

            contours, _ = cv2.findContours(
                cutout_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x_, y_, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                if aspect_ratio > 5:
                    continue

            cx, cy = cutout_img.shape[1] // 2, cutout_img.shape[0] // 2
            cv2.drawMarker(cutout_img, (cx, cy), 255, cv2.MARKER_CROSS, 10, 1)

            out_path = os.path.join(output_dir, f"detection_{i:04}.png")
            cv2.imwrite(out_path, cutout_img)

            ra_deg, dec_deg = None, None
            if wcs_objects[i] is not None:
                try:
                    sky_coords = wcs_objects[i].pixel_to_world(x, y)
                    ra_deg = round(sky_coords.ra.deg, 6)
                    dec_deg = round(sky_coords.dec.deg, 6)
                except Exception:
                    pass

            light_curve = []
            for f in frames:
                y1, y2 = max(0, y - 3), y + 4
                x1, x2 = max(0, x - 3), x + 4
                aperture = f[y1:y2, x1:x2]
                brightness = float(np.sum(aperture))
                light_curve.append(brightness)

            detections.append(
                {
                    "frame": i,
                    "filename": filenames[i],
                    "x": int(x),
                    "y": int(y),
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "z_score": float(z_value),
                    "dimming": z_value < 0,
                    "confidence": round(float(min(1.0, abs(z_value) / 12.0)), 2),
                    "cutout_image": out_path,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "light_curve": light_curve,
                }
            )

    return detections


def crossmatch_with_gaia(detections, radius_arcsec=5.0):
    matched = []
    for det in detections:
        if det["ra_deg"] is None or det["dec_deg"] is None:
            det.update(
                {
                    "match_name": None,
                    "object_type": None,
                    "angular_distance_arcsec": None,
                }
            )
            matched.append(det)
            continue

        coord = SkyCoord(
            ra=det["ra_deg"] * u.deg, dec=det["dec_deg"] * u.deg, frame="icrs"
        )
        try:
            query = f"""
                SELECT TOP 1 source_id, ra, dec, phot_g_mean_mag, parallax
                FROM gaiadr3.gaia_source
                WHERE 1=CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE(
    'ICRS', {
                coord.ra.deg}, {
                coord.dec.deg}, {
                radius_arcsec / 3600.0})
                )
            """
            job = Gaia.launch_job(query)
            result = job.get_results()

            if len(result) > 0:
                r = result[0]
                gaia_coord = SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
                sep = coord.separation(gaia_coord).arcsecond
                parallax = float(r["parallax"])
                distance_pc = round(1000.0 / parallax, 2) if parallax > 0 else -1
                det.update(
                    {
                        "match_name": f"GAIA DR3 {r['source_id']}",
                        "object_type": "Star",
                        "angular_distance_arcsec": round(sep, 2),
                        "g_mag": float(r["phot_g_mean_mag"]),
                        "distance_pc": distance_pc,
                    }
                )
            else:
                det.update(
                    {
                        "match_name": None,
                        "object_type": None,
                        "angular_distance_arcsec": None,
                    }
                )
        except Exception:
            det.update(
                {
                    "match_name": None,
                    "object_type": None,
                    "angular_distance_arcsec": None,
                }
            )

        matched.append(det)
    return matched


def save_report(detections, output_path="pulse_report.json"):
    with open(output_path, "w") as f:
        json.dump({"detections": detections}, f, indent=2)

    try:
        with open(output_path, "r") as f:
            data = f.read()
        response = requests.post(
            "https://geekastro.dev/pulsehunter/submit_report.php", data=data
        )
        if response.ok:
            print("✅ Report uploaded successfully.")
            QMessageBox.information(
                None, "Upload Success", "Report uploaded to geekastro.dev."
            )
        else:
            print("⚠️ Upload failed:", response.text)
            QMessageBox.warning(None, "Upload Failed", response.text)
    except Exception as e:
        print("❌ Upload error:", e)
        QMessageBox.critical(None, "Upload Error", str(e))
