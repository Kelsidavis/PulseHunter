"""
PulseHunter Core Module - Fixed Version
Astronomy transient detection and analysis toolkit
"""

import json

# Simple logging for core module
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from reproject import reproject_interp
from astropy.wcs import WCS

def wcs_align_frames(frames, wcs_list, reference_index=0):
    """
    Reproject all frames onto the WCS grid of the reference frame.
    Args:
        frames: List or array of 2D numpy arrays
        wcs_list: List of WCS objects (same length as frames)
        reference_index: Which frame to use as the alignment reference
    Returns:
        aligned_frames: np.ndarray, shape = (n_frames, height, width)
        reference_wcs: WCS of reference frame
    """
    reference_wcs = wcs_list[reference_index]
    shape_out = frames[reference_index].shape
    aligned_frames = []

    for i, (frame, wcs) in enumerate(zip(frames, wcs_list)):
        if wcs is not None and reference_wcs is not None:
            # Reproject to reference WCS grid
            aligned_data, _ = reproject_interp(
                (frame, wcs), reference_wcs, shape_out=shape_out, order='bilinear'
            )
        else:
            # No WCS—use original
            aligned_data = frame
        aligned_frames.append(aligned_data)
    return np.array(aligned_frames), reference_wcs

logger = logging.getLogger(__name__)


def plate_solve_astap(filepath: str, astap_exe: str = "astap") -> bool:
    """
    Plate solve a FITS file using ASTAP

    Args:
        filepath: Path to FITS file to solve
        astap_exe: Path to ASTAP executable

    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if already has WCS
        with fits.open(filepath, mode="update") as hdul:
            if "CRVAL1" in hdul[0].header and "CRVAL2" in hdul[0].header:
                logger.info(f"File already has WCS: {filepath}")
                return True

        # Run ASTAP
        result = subprocess.run(
            [astap_exe, "-f", filepath, "-update"], capture_output=True, timeout=60
        )

        if result.returncode == 0:
            logger.info(f"ASTAP solved: {filepath}")
            return True
        else:
            logger.warning(f"ASTAP failed for {filepath}: {result.stderr.decode()}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"ASTAP timeout for: {filepath}")
        return False
    except Exception as e:
        logger.error(f"ASTAP error for {filepath}: {e}")
        return False


def load_calibrated_fits(
    folder: str,
    plate_solve_missing: bool = False,
    astap_exe: str = "astap",
    progress_callback: Optional[callable] = None,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[Optional[WCS]]]:
    """
    Load calibrated FITS files from a folder

    Args:
        folder: Directory containing calibrated FITS files
        plate_solve_missing: Whether to plate solve files without WCS
        astap_exe: Path to ASTAP executable
        progress_callback: Optional progress callback function
        max_files: Maximum number of files to load (None for all)

    Returns:
        Tuple of (frames array, filenames list, wcs_objects list)
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder}")
        return np.array([]), [], []

    # Find FITS files
    fits_files = []
    for ext in ["*.fits", "*.fit", "*.fts", "*.FIT", "*.FITS"]:
        fits_files.extend(folder_path.glob(ext))
    fits_files = sorted(fits_files)[:max_files] if max_files else sorted(fits_files)

    if not fits_files:
        logger.error(f"No FITS files found in {folder}")
        return np.array([]), [], []

    logger.info(f"Loading {len(fits_files)} FITS files from {folder}")

    frames = []
    filenames = []
    wcs_objects = []

    # Load files with progress
    for i, file_path in enumerate(fits_files):
        try:
            # Plate solve if requested
            if plate_solve_missing:
                plate_solve_astap(str(file_path), astap_exe)

            # Load FITS data
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

                if data is None:
                    logger.warning(f"No data in {file_path}")
                    continue

                # Convert to float32 for processing
                data = data.astype(np.float32)

                # Create WCS object if available
                wcs = None
                if "CRVAL1" in header and "CRVAL2" in header:
                    try:
                        wcs = WCS(header)
                    except Exception as e:
                        logger.warning(f"Failed to create WCS for {file_path}: {e}")

                frames.append(data)
                filenames.append(str(file_path))
                wcs_objects.append(wcs)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue

        # Update progress
        if progress_callback:
            progress = int((i + 1) / len(fits_files) * 100)
            progress_callback(progress)

    if not frames:
        logger.error("No valid frames loaded")
        return np.array([]), [], []

    logger.info(f"Successfully loaded {len(frames)} frames")
    frames_array = np.array(frames)

    return frames_array, filenames, wcs_objects


def detect_transients(
    frames: np.ndarray,
    filenames: List[str],
    wcs_objects: List[Optional[WCS]],
    output_dir: str = "detections",
    z_thresh: float = 6.0,
    min_pixels: int = 4,
    edge_margin: int = 20,
    detect_dimming: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[Dict]:
    """
    Detect transient objects in a stack of astronomical images

    Args:
        frames: Array of image frames
        filenames: List of corresponding filenames
        wcs_objects: List of WCS objects for coordinate conversion
        output_dir: Directory to save detection info
        z_thresh: Z-score threshold for detection
        min_pixels: Minimum connected pixels for valid detection
        edge_margin: Margin from image edges to ignore
        detect_dimming: Whether to detect dimming events
        progress_callback: Optional progress callback

    Returns:
        List of detection dictionaries
    """
    if len(frames) < 3:
        logger.error("Need at least 3 frames for transient detection")
        return []

    logger.info(f"Analyzing {len(frames)} frames for transients (z>{z_thresh})")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Calculate median and MAD (more robust than mean/std)
    median_image = np.median(frames, axis=0)
    mad_image = np.median(np.abs(frames - median_image), axis=0)

    # Convert MAD to equivalent standard deviation
    std_image = 1.4826 * mad_image

    # Avoid division by zero
    std_image = np.where(std_image > 0, std_image, 1.0)

    detections = []

    # Process each frame
    for i, (frame, filename) in enumerate(zip(frames, filenames)):
        # Calculate z-scores
        z_scores = (frame - median_image) / std_image

        # Find significant pixels
        if detect_dimming:
            significant = np.abs(z_scores) > z_thresh
        else:
            significant = z_scores > z_thresh

        # Remove edge pixels
        significant[:edge_margin, :] = False
        significant[-edge_margin:, :] = False
        significant[:, :edge_margin] = False
        significant[:, -edge_margin:] = False

        # Find connected regions
        from scipy import ndimage

        labeled, num_features = ndimage.label(significant)

        # Process each detection
        for label_num in range(1, num_features + 1):
            region_mask = labeled == label_num
            region_pixels = np.sum(region_mask)

            # Skip small regions
            if region_pixels < min_pixels:
                continue

            # Get region properties
            y_coords, x_coords = np.where(region_mask)
            y_center = int(np.mean(y_coords))
            x_center = int(np.mean(x_coords))

            # Get peak z-score in region
            region_z_scores = z_scores[region_mask]
            if detect_dimming:
                peak_idx = np.argmax(np.abs(region_z_scores))
                peak_z = region_z_scores[peak_idx]
            else:
                peak_z = np.max(region_z_scores)

            # Calculate coordinates if WCS available
            ra_deg = None
            dec_deg = None
            if i < len(wcs_objects) and wcs_objects[i] is not None:
                try:
                    sky_coord = wcs_objects[i].pixel_to_world(x_center, y_center)
                    ra_deg = float(sky_coord.ra.deg)
                    dec_deg = float(sky_coord.dec.deg)
                except Exception as e:
                    logger.warning(f"WCS conversion failed: {e}")

            # Extract light curve at this position
            light_curve = []
            aperture_radius = 3
            for f_idx, f in enumerate(frames):
                y1 = max(0, y_center - aperture_radius)
                y2 = min(f.shape[0], y_center + aperture_radius + 1)
                x1 = max(0, x_center - aperture_radius)
                x2 = min(f.shape[1], x_center + aperture_radius + 1)

                aperture_sum = float(np.sum(f[y1:y2, x1:x2]))
                light_curve.append(aperture_sum)

            # Create detection record
            detection = {
                "frame": i,
                "filename": filename,
                "x": int(x_center),
                "y": int(y_center),
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "z_score": float(peak_z),
                "confidence": float(min(1.0, abs(peak_z) / z_thresh)),
                "dimming": bool(peak_z < 0),
                "pixels": int(region_pixels),
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "light_curve": light_curve,
            }

            detections.append(detection)
            logger.info(
                f"Detection in frame {i}: z={peak_z:.1f} at ({x_center},{y_center})"
            )

        # Update progress
        if progress_callback:
            progress = int((i + 1) / len(frames) * 100)
            progress_callback(progress)

    logger.info(f"Found {len(detections)} detections")
    return detections


def save_detection_report(
    detections: List[Dict],
    output_path: str = "detection_report.json",
    metadata: Optional[Dict] = None,
) -> bool:
    """
    Save detection report to JSON file

    Args:
        detections: List of detection dictionaries
        output_path: Output file path
        metadata: Optional metadata to include

    Returns:
        True if successful
    """
    try:
        report = {
            "detections": detections,
            "metadata": metadata or {},
            "summary": {
                "total_detections": len(detections),
                "dimming_events": sum(1 for d in detections if d.get("dimming")),
                "brightening_events": sum(
                    1 for d in detections if not d.get("dimming")
                ),
                "with_coordinates": sum(
                    1 for d in detections if d.get("ra_deg") is not None
                ),
                "high_confidence": sum(
                    1 for d in detections if d.get("confidence", 0) > 0.8
                ),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        }

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=convert_numpy)

        logger.info(f"Saved detection report to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return False


def create_detection_images(
    detections: List[Dict],
    frames: np.ndarray,
    output_dir: str = "detection_images",
    image_size: int = 100,
) -> None:
    """
    Create cutout images for each detection

    Args:
        detections: List of detections
        frames: Array of frames
        output_dir: Output directory for images
        image_size: Size of cutout images
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for det in detections:
        try:
            frame_idx = det["frame"]
            if frame_idx >= len(frames):
                continue

            frame = frames[frame_idx]
            x, y = det["x"], det["y"]

            # Extract cutout
            half_size = image_size // 2
            y1 = max(0, y - half_size)
            y2 = min(frame.shape[0], y + half_size)
            x1 = max(0, x - half_size)
            x2 = min(frame.shape[1], x + half_size)

            cutout = frame[y1:y2, x1:x2]

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Detection image
            vmin, vmax = np.percentile(cutout, [1, 99])
            im1 = ax1.imshow(cutout, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

            # Mark detection center
            center_x = x - x1
            center_y = y - y1
            circle = Circle(
                (center_x, center_y), 5, color="red", fill=False, linewidth=2
            )
            ax1.add_patch(circle)

            ax1.set_title(f'Detection: z={det["z_score"]:.1f}')
            plt.colorbar(im1, ax=ax1)

            # Light curve
            if "light_curve" in det and det["light_curve"]:
                frames_range = range(len(det["light_curve"]))
                ax2.plot(frames_range, det["light_curve"], "b-o", markersize=4)
                ax2.axvline(frame_idx, color="red", linestyle="--", label="Detection")
                ax2.set_xlabel("Frame Number")
                ax2.set_ylabel("Brightness (ADU)")
                ax2.set_title("Light Curve")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

            # Save figure
            output_file = output_path / f'detection_{det["frame"]:04d}_{x}_{y}.png'
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Saved detection image: {output_file.name}")

        except Exception as e:
            logger.error(f"Failed to create detection image: {e}")
            continue
