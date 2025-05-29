"""
Updated PulseHunter Core Module with Enhanced Calibration Integration
Astronomy transient detection and analysis toolkit
"""

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

# Import Qt components with safety check
try:
    from PySide6.QtWidgets import QApplication, QMessageBox

    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Import enhanced calibration with fallback
try:
    from calibration_integration import (
        get_calibration_info,
        print_calibration_status,
        smart_load_fits_stack,
    )

    ENHANCED_CALIBRATION_AVAILABLE = True
    print("✅ Enhanced calibration system loaded")
except ImportError as e:
    ENHANCED_CALIBRATION_AVAILABLE = False
    print(f"⚠️ Enhanced calibration not available: {e}")


def _show_message_box(title, message, msg_type="info"):
    """Safely show message box only if Qt is available and app exists"""
    if not QT_AVAILABLE:
        return False

    try:
        app = QApplication.instance()
        if app is None:
            # No Qt application running, can't show dialogs
            return False

        if msg_type == "info":
            QMessageBox.information(None, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(None, title, message)
        elif msg_type == "error":
            QMessageBox.critical(None, title, message)
        return True
    except Exception:
        # Qt not properly initialized or other GUI error
        return False


def plate_solve_astap(filepath, astap_exe="astap"):
    """
    Plate solve a FITS file using ASTAP

    Args:
        filepath (str): Path to FITS file to solve
        astap_exe (str): Path to ASTAP executable
    """
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
    except subprocess.TimeoutExpired:
        print(f"ASTAP timeout for: {filepath}")
    except FileNotFoundError:
        print(f"ASTAP executable not found: {astap_exe}")
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
    progress_callback=None,
    auto_calibrate=True,
    **kwargs,
):
    """
    Enhanced FITS stack loading with automatic calibration detection

    This function now uses the enhanced calibration system when available,
    with fallback to basic loading for compatibility.

    Args:
        folder (str): Directory containing FITS files
        plate_solve_missing (bool): Whether to plate solve files without WCS
        astap_exe (str): Path to ASTAP executable
        master_bias (np.ndarray): Master bias frame for calibration
        master_dark (np.ndarray): Master dark frame for calibration
        master_flat (np.ndarray): Master flat frame for calibration
        camera_mode (str): Camera mode ("mono" or "osc")
        filter_name (str): Filter name for the observation
        progress_callback (callable): Optional progress callback function
        auto_calibrate (bool): Whether to automatically detect and apply calibration
        **kwargs: Additional arguments

    Returns:
        tuple: (frames array, filenames list, wcs_objects list)
    """
    print(f"🔄 Loading FITS stack from: {folder}")

    # Use enhanced loading if available
    if ENHANCED_CALIBRATION_AVAILABLE:
        try:
            print("✨ Using enhanced calibration system...")

            # Show calibration status if requested
            if auto_calibrate:
                print_calibration_status(folder)

            return smart_load_fits_stack(
                folder=folder,
                plate_solve_missing=plate_solve_missing,
                astap_exe=astap_exe,
                auto_calibrate=auto_calibrate,
                manual_master_bias=master_bias,
                manual_master_dark=master_dark,
                manual_master_flat=master_flat,
                progress_callback=progress_callback,
                **kwargs,
            )
        except Exception as e:
            print(f"⚠️ Enhanced loading failed: {e}")
            print("Falling back to basic loading...")

    # Fallback to basic loading
    return _basic_load_fits_stack(
        folder=folder,
        plate_solve_missing=plate_solve_missing,
        astap_exe=astap_exe,
        master_bias=master_bias,
        master_dark=master_dark,
        master_flat=master_flat,
        camera_mode=camera_mode,
        filter_name=filter_name,
        **kwargs,
    )


def _basic_load_fits_stack(
    folder,
    plate_solve_missing=False,
    astap_exe="astap",
    master_bias=None,
    master_dark=None,
    master_flat=None,
    camera_mode="mono",
    filter_name=None,
):
    """
    Basic FITS stack loading (original implementation)
    """
    from concurrent.futures import ThreadPoolExecutor

    frames, filenames, wcs_objects = [], [], []

    if not os.path.exists(folder):
        print(f"Error: Folder {folder} does not exist")
        return np.array([]), [], []

    fits_files = [
        f for f in sorted(os.listdir(folder)) if f.endswith((".fits", ".fit", ".fts"))
    ]
    if not fits_files:
        print(f"No FITS files found in {folder}")
        return np.array([]), [], []

    print(f"📁 Basic loading of {len(fits_files)} FITS files...")

    def load_fits_file(file):
        path = os.path.join(folder, file)
        try:
            # Check for existing WCS
            hdr = fits.getheader(path)
            has_wcs = "CRVAL1" in hdr and "CRVAL2" in hdr

            # Plate solve if requested
            if plate_solve_missing and not has_wcs:
                plate_solve_astap(path, astap_exe)

            # Load data
            hdul = fits.open(path)
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            hdul.close()

            # Apply calibration frames if provided
            if master_bias is not None:
                data -= master_bias
            if master_dark is not None:
                data -= master_dark
            if master_flat is not None and np.all(master_flat != 0):
                data /= master_flat

            # Create WCS object
            wcs = WCS(header) if "CRVAL1" in header and "CRVAL2" in header else None

            return (data, file, wcs)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return None

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(filter(None, executor.map(load_fits_file, fits_files)))

    if not results:
        return np.array([]), [], []

    frames, filenames, wcs_objects = zip(*results)

    print(f"✅ Loaded {len(frames)} frames")
    return np.array(frames), list(filenames), list(wcs_objects)


# Enhanced version that integrates with calibration system
def enhanced_load_fits_stack(
    folder,
    plate_solve_missing=True,
    astap_exe="astap",
    auto_calibrate=True,
    manual_master_bias=None,
    manual_master_dark=None,
    manual_master_flat=None,
    progress_callback=None,
    **kwargs,
):
    """
    Enhanced FITS loading with automatic calibration (alias for compatibility)

    This is an alias to the smart_load_fits_stack function from the calibration
    integration module, maintaining compatibility with existing code.
    """
    if ENHANCED_CALIBRATION_AVAILABLE:
        return smart_load_fits_stack(
            folder=folder,
            plate_solve_missing=plate_solve_missing,
            astap_exe=astap_exe,
            auto_calibrate=auto_calibrate,
            manual_master_bias=manual_master_bias,
            manual_master_dark=manual_master_dark,
            manual_master_flat=manual_master_flat,
            progress_callback=progress_callback,
            **kwargs,
        )
    else:
        # Fall back to basic loading
        return load_fits_stack(
            folder=folder,
            plate_solve_missing=plate_solve_missing,
            astap_exe=astap_exe,
            master_bias=manual_master_bias,
            master_dark=manual_master_dark,
            master_flat=manual_master_flat,
            **kwargs,
        )


def detect_transients(
    frames,
    filenames,
    wcs_objects,
    output_dir="detections",
    z_thresh=6.0,
    cutout_size=50,
    edge_margin=20,
    detect_dimming=False,
    progress_callback=None,
):
    """
    Detect transient objects in a stack of astronomical images
    Enhanced with progress reporting

    Args:
        frames (np.ndarray): Array of image frames
        filenames (list): List of corresponding filenames
        wcs_objects (list): List of WCS objects for coordinate conversion
        output_dir (str): Directory to save detection cutouts
        z_thresh (float): Z-score threshold for detection
        cutout_size (int): Size of cutout images in pixels
        edge_margin (int): Margin from image edges to ignore
        detect_dimming (bool): Whether to detect dimming events
        progress_callback (callable): Optional progress callback

    Returns:
        list: List of detection dictionaries
    """
    from concurrent.futures import ThreadPoolExecutor

    if len(frames) == 0:
        print("No frames loaded.")
        return []

    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing {len(frames)} frames for transients...")

    # Update progress
    if progress_callback:
        progress_callback(10)

    # Calculate statistical reference images
    mean_image = np.mean(frames, axis=0)
    std_image = np.std(frames, axis=0)

    if progress_callback:
        progress_callback(30)

    def process_single_frame(i):
        frame = frames[i]
        z = (frame - mean_image) / (std_image + 1e-5)
        y, x = np.unravel_index(np.argmax(np.abs(z)), z.shape)
        value = z[y, x]

        if (
            abs(value) > z_thresh
            and edge_margin < x < z.shape[1] - edge_margin
            and edge_margin < y < z.shape[0] - edge_margin
        ):
            # Create cutout image
            y1 = max(0, y - cutout_size // 2)
            y2 = min(frame.shape[0], y + cutout_size // 2)
            x1 = max(0, x - cutout_size // 2)
            x2 = min(frame.shape[1], x + cutout_size // 2)

            cutout = frame[y1:y2, x1:x2]

            # Normalize cutout for display
            cutout_min, cutout_max = np.min(cutout), np.max(cutout)
            if cutout_max > cutout_min:
                cutout_norm = 255 * (cutout - cutout_min) / (cutout_max - cutout_min)
            else:
                cutout_norm = np.zeros_like(cutout)
            cutout_img = cutout_norm.astype(np.uint8)

            # Add crosshair marker
            cx, cy = cutout_img.shape[1] // 2, cutout_img.shape[0] // 2
            cv2.drawMarker(cutout_img, (cx, cy), 255, cv2.MARKER_CROSS, 10, 1)

            # Save cutout
            out_path = os.path.join(output_dir, f"detection_{i:04d}.png")
            cv2.imwrite(out_path, cutout_img)

            # Calculate celestial coordinates
            wcs = wcs_objects[i]
            ra_deg, dec_deg = (None, None)
            if wcs and wcs.has_celestial:
                try:
                    sky_coords = wcs.pixel_to_world(x, y)
                    ra_deg = round(sky_coords.ra.deg, 6)
                    dec_deg = round(sky_coords.dec.deg, 6)
                except:
                    pass

            # Generate light curve
            light_curve = []
            aperture_radius = 3
            for f in frames:
                y1 = max(0, y - aperture_radius)
                y2 = min(f.shape[0], y + aperture_radius + 1)
                x1 = max(0, x - aperture_radius)
                x2 = min(f.shape[1], x + aperture_radius + 1)
                aperture = f[y1:y2, x1:x2]
                brightness = float(np.sum(aperture))
                light_curve.append(brightness)

            return {
                "frame": i,
                "filename": filenames[i],
                "x": int(x),
                "y": int(y),
                "ra_deg": float(ra_deg) if ra_deg else None,
                "dec_deg": float(dec_deg) if dec_deg else None,
                "z_score": float(value),
                "confidence": float(min(1.0, abs(value) / z_thresh)),
                "dimming": bool(value < 0 and detect_dimming),
                "cutout_image": out_path,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "light_curve": light_curve,
            }
        return None

    # Process frames in parallel with progress updates
    detections = []
    with ThreadPoolExecutor() as executor:
        for i, result in enumerate(
            executor.map(process_single_frame, range(len(frames)))
        ):
            if result is not None:
                detections.append(result)
                print(
                    f"Detection {len(detections)}: z={result['z_score']:.1f} at ({result['x']},{result['y']})"
                )

            # Update progress
            if progress_callback:
                progress = 30 + int((i / len(frames)) * 50)
                progress_callback(progress)

    if progress_callback:
        progress_callback(80)

    print(f"Found {len(detections)} detections")
    return detections


def crossmatch_with_gaia(detections, radius_arcsec=5.0, progress_callback=None):
    """
    Cross-match detections with Gaia DR3 catalog
    Enhanced with progress reporting

    Args:
        detections (list): List of detection dictionaries
        radius_arcsec (float): Search radius in arcseconds
        progress_callback (callable): Optional progress callback

    Returns:
        list: Updated detections with Gaia matches
    """
    matched = []

    print(f"Cross-matching {len(detections)} detections with Gaia DR3...")

    for i, det in enumerate(detections):
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
            # Query Gaia DR3 for nearby sources
            query = f"""
                SELECT TOP 1 source_id, ra, dec, phot_g_mean_mag, parallax
                FROM gaiadr3.gaia_source
                WHERE 1=CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE(
                        'ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_arcsec / 3600.0}
                    )
                )
                ORDER BY phot_g_mean_mag ASC
            """

            job = Gaia.launch_job(query)
            result = job.get_results()

            if len(result) > 0:
                r = result[0]
                gaia_coord = SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
                sep = coord.separation(gaia_coord).arcsecond

                # Calculate distance from parallax
                parallax = float(r["parallax"]) if r["parallax"] else 0
                distance_pc = round(1000.0 / parallax, 2) if parallax > 0 else -1

                det.update(
                    {
                        "match_name": f"GAIA DR3 {r['source_id']}",
                        "object_type": "Star",
                        "angular_distance_arcsec": round(sep, 2),
                        "g_mag": float(r["phot_g_mean_mag"])
                        if r["phot_g_mean_mag"]
                        else None,
                        "distance_pc": distance_pc,
                    }
                )
                print(f'Detection {i+1}: Matched with Gaia source (sep={sep:.1f}")')
            else:
                det.update(
                    {
                        "match_name": None,
                        "object_type": None,
                        "angular_distance_arcsec": None,
                    }
                )
                print(f"Detection {i+1}: No Gaia match found")

        except Exception as e:
            print(f"Gaia query failed for detection {i+1}: {e}")
            det.update(
                {
                    "match_name": None,
                    "object_type": None,
                    "angular_distance_arcsec": None,
                }
            )

        matched.append(det)

        # Update progress
        if progress_callback:
            progress = int((i / len(detections)) * 100)
            progress_callback(progress)

    return matched


def save_report(detections, output_path="pulse_report.json"):
    """
    Save detection report with improved upload handling

    Args:
        detections (list): List of detection dictionaries
        output_path (str): Path to save the JSON report

    Returns:
        bool: True if local save successful, False otherwise
    """
    try:
        # Prepare report data
        report_data = {
            "detections": detections,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_detections": len(detections),
                "pulsehunter_version": "Enhanced-1.0.0",
                "high_confidence_count": sum(
                    1 for d in detections if d.get("confidence", 0) > 0.8
                ),
                "exoplanet_candidates": sum(
                    1 for d in detections if d.get("exo_match")
                ),
                "gaia_matches": sum(1 for d in detections if d.get("match_name")),
                "enhanced_calibration": ENHANCED_CALIBRATION_AVAILABLE,
            },
        }

        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Save local file first (this is the important part)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=convert_numpy)

        print(f"✅ Report saved locally: {output_path}")

        # Try upload but don't fail if it doesn't work
        upload_success = attempt_upload(report_data, output_path)

        return True  # Return True as long as local save worked

    except IOError as e:
        error_msg = f"File save error: {e}"
        print(f"❌ {error_msg}")
        _show_message_box("Save Error", error_msg, "error")
        return False
    except Exception as e:
        error_msg = f"Unexpected error in save_report: {e}"
        print(f"❌ {error_msg}")
        _show_message_box("Error", error_msg, "error")
        return False


def attempt_upload(report_data, local_path):
    """
    Attempt to upload report to various endpoints

    Returns:
        bool: True if any upload succeeded
    """
    upload_endpoints = [
        "https://geekastro.dev/pulsehunter/submit_report.php",
        "https://api.geekastro.dev/pulsehunter/submit",
        # Add more backup endpoints here if needed
    ]

    for endpoint in upload_endpoints:
        try:
            print(f"🔄 Attempting upload to {endpoint}...")

            # Prepare data for upload
            with open(local_path, "r") as f:
                data = f.read()

            response = requests.post(
                endpoint,
                data=data,
                timeout=15,  # Reduced timeout
                headers={"Content-Type": "application/json"},
            )

            if response.ok:
                print(f"✅ Report uploaded successfully to {endpoint}")
                _show_message_box(
                    "Upload Success",
                    f"Report uploaded to {endpoint.split('/')[2]}",
                    "info",
                )
                return True
            else:
                print(
                    f"⚠️ Upload failed to {endpoint}: {response.status_code} - {response.text[:100]}"
                )
                continue

        except requests.Timeout:
            print(f"⚠️ Upload timeout to {endpoint}")
            continue
        except requests.ConnectionError:
            print(f"⚠️ Connection error to {endpoint} - server may be offline")
            continue
        except requests.RequestException as e:
            print(f"⚠️ Network error to {endpoint}: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Unexpected error uploading to {endpoint}: {e}")
            continue

    # If all uploads failed, show informative message
    print("⚠️ All upload attempts failed - results saved locally only")
    _show_message_box(
        "Upload Note",
        "✅ Results saved locally!\n\n"
        "Online upload failed - this is normal if the server is not available.\n"
        "Your detection data is safely saved on your computer.",
        "warning",
    )
    return False


def generate_summary_stats(detections):
    """
    Generate summary statistics for detections

    Args:
        detections (list): List of detection dictionaries

    Returns:
        dict: Summary statistics
    """
    if not detections:
        return {"total": 0}

    stats = {
        "total": len(detections),
        "with_coordinates": sum(1 for d in detections if d.get("ra_deg")),
        "gaia_matches": sum(1 for d in detections if d.get("match_name")),
        "dimming_events": sum(1 for d in detections if d.get("dimming")),
        "brightening_events": sum(1 for d in detections if not d.get("dimming", True)),
        "avg_confidence": round(
            np.mean([d.get("confidence", 0) for d in detections]), 2
        ),
        "max_z_score": round(max([abs(d.get("z_score", 0)) for d in detections]), 1),
        "enhanced_calibration_used": ENHANCED_CALIBRATION_AVAILABLE,
    }

    return stats


# New utility functions for the enhanced system
def validate_calibration_for_folder(lights_folder):
    """
    Validate calibration setup for a specific folder

    Args:
        lights_folder (str): Path to lights folder

    Returns:
        dict: Validation results
    """
    if ENHANCED_CALIBRATION_AVAILABLE:
        try:
            return get_calibration_info(lights_folder)
        except Exception as e:
            return {"error": f"Calibration validation failed: {e}"}
    else:
        return {
            "enhanced_available": False,
            "message": "Enhanced calibration not available - using basic calibration",
        }


def print_system_status():
    """Print comprehensive system status"""
    print(f"\n{'='*60}")
    print("PULSEHUNTER SYSTEM STATUS")
    print(f"{'='*60}")

    print(
        f"Enhanced Calibration: {'✅ Available' if ENHANCED_CALIBRATION_AVAILABLE else '❌ Not Available'}"
    )
    print(f"Qt GUI Support: {'✅ Available' if QT_AVAILABLE else '❌ Not Available'}")

    # Test key dependencies
    dependencies = {
        "Astropy": "astropy",
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "Requests": "requests",
        "AstroQuery": "astroquery",
    }

    print(f"\nDependencies:")
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")

    print(f"\nAvailable Functions:")
    print(
        f"  • load_fits_stack() - {'Enhanced' if ENHANCED_CALIBRATION_AVAILABLE else 'Basic'} FITS loading"
    )
    print(f"  • detect_transients() - Transient detection with progress tracking")
    print(f"  • crossmatch_with_gaia() - GAIA catalog cross-matching")
    print(f"  • save_report() - Report generation and upload")

    if ENHANCED_CALIBRATION_AVAILABLE:
        print(f"  • validate_calibration_for_folder() - Calibration validation")
        print(f"  • enhanced_load_fits_stack() - Advanced calibration loading")


if __name__ == "__main__":
    # Print system status when run directly
    print("🌌 PulseHunter Core Module - Enhanced Version")
    print_system_status()

    # Test enhanced calibration if available
    if ENHANCED_CALIBRATION_AVAILABLE:
        print(f"\n{'='*60}")
        print("ENHANCED CALIBRATION TEST")
        print(f"{'='*60}")

        test_folder = r"F:\astrophotography\2024-07-13 - bortle2 - sthelens"
        if os.path.exists(test_folder):
            print(f"Testing calibration with: {test_folder}")
            validation = validate_calibration_for_folder(test_folder)

            if "error" not in validation:
                print(f"✅ Calibration validation completed")
                print(f"   Status: {validation.get('calibration_status', 'unknown')}")
                print(f"   Master files: {len(validation.get('master_files', {}))}")
            else:
                print(f"❌ Validation failed: {validation['error']}")
        else:
            print("Test folder not found - update path to test with your data")

    print(f"\n✅ PulseHunter Core ready for enhanced astronomical processing!")
