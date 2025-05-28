"""
PulseHunter Core Module
Astronomy transient detection and analysis toolkit
Updated with improved upload handling and error recovery
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
):
    """
    Load and optionally calibrate a stack of FITS files
    
    Args:
        folder (str): Directory containing FITS files
        plate_solve_missing (bool): Whether to plate solve files without WCS
        astap_exe (str): Path to ASTAP executable
        master_bias (np.ndarray): Master bias frame for calibration
        master_dark (np.ndarray): Master dark frame for calibration
        master_flat (np.ndarray): Master flat frame for calibration
        camera_mode (str): Camera mode ("mono" or "osc")
        filter_name (str): Filter name for the observation
        
    Returns:
        tuple: (frames array, filenames list, wcs_objects list)
    """
    frames = []
    filenames = []
    wcs_objects = []

    if not os.path.exists(folder):
        print(f"Error: Folder {folder} does not exist")
        return np.array([]), [], []

    fits_files = [f for f in sorted(os.listdir(folder)) if f.endswith(".fits")]
    if not fits_files:
        print(f"No FITS files found in {folder}")
        return np.array([]), [], []

    for file in fits_files:
        path = os.path.join(folder, file)

        try:
            # Check for existing WCS
            hdr = fits.getheader(path)
            has_wcs = "CRVAL1" in hdr and "CRVAL2" in hdr
        except Exception as e:
            print(f"Error reading header from {file}: {e}")
            has_wcs = False

        # Plate solve if requested and no WCS exists
        if plate_solve_missing and not has_wcs:
            plate_solve_astap(path, astap_exe)

        try:
            # Load FITS data
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

            # Handle different camera modes
            if camera_mode == "osc":
                pass  # Reserved for future OSC (One Shot Color) handling

            frames.append(data)
            filenames.append(file)

            # Extract WCS information
            try:
                wcs = WCS(header)
                if not wcs.has_celestial:
                    wcs = None
            except Exception:
                wcs = None

            wcs_objects.append(wcs)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if not frames:
        print("No valid FITS files loaded")
        return np.array([]), [], []

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
    """
    Detect transient objects in a stack of astronomical images
    
    Args:
        frames (np.ndarray): Array of image frames
        filenames (list): List of corresponding filenames
        wcs_objects (list): List of WCS objects for coordinate conversion
        output_dir (str): Directory to save detection cutouts
        z_thresh (float): Z-score threshold for detection
        cutout_size (int): Size of cutout images in pixels
        edge_margin (int): Margin from image edges to ignore
        detect_dimming (bool): Whether to detect dimming events
        
    Returns:
        list: List of detection dictionaries
    """
    if len(frames) == 0:
        print("No frames loaded.")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate statistical reference images
    print(f"Analyzing {len(frames)} frames for transients...")
    mean_image = np.mean(frames, axis=0)
    std_image = np.std(frames, axis=0)

    detections = []

    for i, frame in enumerate(frames):
        # Calculate z-score map
        z = (frame - mean_image) / (std_image + 1e-5)
        
        # Find peak deviation
        y, x = np.unravel_index(np.argmax(np.abs(z)), z.shape)
        z_value = z[y, x]

        # Check if detection meets threshold
        if abs(z_value) > z_thresh:
            # Skip detections near image edges
            if (
                x < edge_margin
                or y < edge_margin
                or x > frame.shape[1] - edge_margin
                or y > frame.shape[0] - edge_margin
            ):
                continue

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

            # Basic shape analysis using contours
            try:
                contours, _ = cv2.findContours(
                    cutout_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    x_, y_, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                    # Skip highly elongated objects (likely artifacts)
                    if aspect_ratio > 5:
                        continue
            except Exception:
                pass  # Continue if contour analysis fails

            # Add crosshair marker to cutout
            cx, cy = cutout_img.shape[1] // 2, cutout_img.shape[0] // 2
            cv2.drawMarker(cutout_img, (cx, cy), 255, cv2.MARKER_CROSS, 10, 1)

            # Save cutout image
            out_path = os.path.join(output_dir, f"detection_{i:04d}.png")
            cv2.imwrite(out_path, cutout_img)

            # Calculate celestial coordinates if WCS available
            ra_deg, dec_deg = None, None
            if wcs_objects[i] is not None:
                try:
                    sky_coords = wcs_objects[i].pixel_to_world(x, y)
                    ra_deg = round(sky_coords.ra.deg, 6)
                    dec_deg = round(sky_coords.dec.deg, 6)
                except Exception as e:
                    print(f"WCS conversion failed for detection {i}: {e}")

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

            # Create detection record
            detection = {
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

            detections.append(detection)
            print(f"Detection {len(detections)}: z={z_value:.1f} at ({x},{y})")

    print(f"Found {len(detections)} detections")
    return detections


def crossmatch_with_gaia(detections, radius_arcsec=5.0):
    """
    Cross-match detections with Gaia DR3 catalog
    
    Args:
        detections (list): List of detection dictionaries
        radius_arcsec (float): Search radius in arcseconds
        
    Returns:
        list: Updated detections with Gaia matches
    """
    matched = []
    
    print(f"Cross-matching {len(detections)} detections with Gaia DR3...")
    
    for i, det in enumerate(detections):
        if det["ra_deg"] is None or det["dec_deg"] is None:
            det.update({
                "match_name": None,
                "object_type": None,
                "angular_distance_arcsec": None,
            })
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
                
                det.update({
                    "match_name": f"GAIA DR3 {r['source_id']}",
                    "object_type": "Star",
                    "angular_distance_arcsec": round(sep, 2),
                    "g_mag": float(r["phot_g_mean_mag"]) if r["phot_g_mean_mag"] else None,
                    "distance_pc": distance_pc,
                })
                print(f"Detection {i+1}: Matched with Gaia source (sep={sep:.1f}\")")
            else:
                det.update({
                    "match_name": None,
                    "object_type": None,
                    "angular_distance_arcsec": None,
                })
                print(f"Detection {i+1}: No Gaia match found")
                
        except Exception as e:
            print(f"Gaia query failed for detection {i+1}: {e}")
            det.update({
                "match_name": None,
                "object_type": None,
                "angular_distance_arcsec": None,
            })

        matched.append(det)
        
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
                "pulsehunter_version": "1.0.0",
                "high_confidence_count": sum(1 for d in detections if d.get("confidence", 0) > 0.8),
                "exoplanet_candidates": sum(1 for d in detections if d.get("exo_match")),
                "gaia_matches": sum(1 for d in detections if d.get("match_name"))
            }
        }
        
        import numpy as np

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
                headers={'Content-Type': 'application/json'}
            )
            
            if response.ok:
                print(f"✅ Report uploaded successfully to {endpoint}")
                _show_message_box(
                    "Upload Success", 
                    f"Report uploaded to {endpoint.split('/')[2]}", 
                    "info"
                )
                return True
            else:
                print(f"⚠️ Upload failed to {endpoint}: {response.status_code} - {response.text[:100]}")
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
        "warning"
    )
    return False


def test_upload_connectivity():
    """
    Test if upload servers are reachable
    
    Returns:
        dict: Status of each upload endpoint
    """
    endpoints = [
        "https://geekastro.dev/pulsehunter/submit_report.php",
        "https://api.geekastro.dev/pulsehunter/submit",
    ]
    
    results = {}
    
    for endpoint in endpoints:
        try:
            # Try a simple HEAD request first
            response = requests.head(endpoint, timeout=5)
            if response.status_code in [200, 405]:  # 405 = Method Not Allowed is OK for HEAD
                results[endpoint] = "✅ Reachable"
            else:
                results[endpoint] = f"❌ HTTP {response.status_code}"
        except requests.ConnectionError:
            results[endpoint] = "❌ Connection failed"
        except requests.Timeout:
            results[endpoint] = "❌ Timeout"
        except Exception as e:
            results[endpoint] = f"❌ Error: {e}"
    
    return results


def debug_upload_issue():
    """Debug upload connectivity issues"""
    print("🔍 Debugging upload connectivity...")
    
    connectivity = test_upload_connectivity()
    
    print("\n📡 Upload Server Status:")
    for endpoint, status in connectivity.items():
        print(f"  {endpoint}")
        print(f"    Status: {status}")
    
    # Test basic internet connectivity
    try:
        response = requests.get("https://httpbin.org/get", timeout=5)
        if response.ok:
            print("  ✅ Internet connection: Working")
        else:
            print("  ❌ Internet connection: Issues detected")
    except Exception as e:
        print(f"  ❌ Internet connection: {e}")
    
    print("\n💡 If uploads are failing:")
    print("  1. Check your internet connection")
    print("  2. The server may be temporarily offline")
    print("  3. Your results are still saved locally!")
    print("  4. You can manually share your results files")


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
        "avg_confidence": round(np.mean([d.get("confidence", 0) for d in detections]), 2),
        "max_z_score": round(max([abs(d.get("z_score", 0)) for d in detections]), 1),
    }
    
    return stats


# Additional utility functions for the complete module

def validate_detection_data(detections):
    """Validate detection data structure"""
    required_fields = ["frame", "filename", "x", "y", "z_score"]
    
    for i, det in enumerate(detections):
        for field in required_fields:
            if field not in det:
                print(f"Warning: Detection {i} missing required field '{field}'")
                return False
    return True


def filter_detections(detections, min_confidence=0.5, max_gaia_distance=2.0):
    """Filter detections based on confidence and Gaia matching"""
    filtered = []
    
    for det in detections:
        # Check confidence threshold
        if det.get("confidence", 0) < min_confidence:
            continue
            
        # Check Gaia distance if matched
        gaia_dist = det.get("angular_distance_arcsec")
        if gaia_dist is not None and gaia_dist > max_gaia_distance:
            continue
            
        filtered.append(det)
    
    return filtered


if __name__ == "__main__":
    # Example usage and testing
    print("PulseHunter Core Module loaded successfully")
    print("Available functions:")
    print("- load_fits_stack()")
    print("- detect_transients()")
    print("- crossmatch_with_gaia()")
    print("- save_report()")
    print("- plate_solve_astap()")
    print("- debug_upload_issue()")
    
    # Test upload connectivity if run directly
    print("\nTesting upload connectivity...")
    debug_upload_issue()