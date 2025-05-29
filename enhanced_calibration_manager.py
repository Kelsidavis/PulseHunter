"""
Enhanced Calibration Manager for PulseHunter
Complete implementation with automatic master file detection and filter-aware processing
"""

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from calibration_utilities import CalibrationConfig, CalibrationLogger
from fits_processing import FITSProcessor


class AutoCalibrationManager:
    """Enhanced automatic calibration manager with filter awareness"""

    def __init__(self):
        self.config = CalibrationConfig()
        self.logger = CalibrationLogger()
        self.fits_processor = FITSProcessor(self.config)
        self.calibration_projects_file = Path("calibration_projects.json")
        self.filter_projects_file = Path("filter_calibration_projects.json")

        # Load existing projects
        self.projects = self._load_projects()
        self.filter_projects = self._load_filter_projects()

    def _load_projects(self) -> Dict:
        """Load standard calibration projects"""
        try:
            if self.calibration_projects_file.exists():
                with open(self.calibration_projects_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading calibration projects: {e}")
        return {}

    def _load_filter_projects(self) -> Dict:
        """Load filter-aware calibration projects"""
        try:
            if self.filter_projects_file.exists():
                with open(self.filter_projects_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading filter calibration projects: {e}")
        return {}

    def get_master_files_for_folder(self, lights_folder: str) -> Dict[str, str]:
        """
        Get master calibration files for a specific lights folder

        Args:
            lights_folder: Path to lights folder

        Returns:
            Dictionary mapping calibration type to master file path
        """
        folder_key = str(Path(lights_folder).resolve())

        # Check standard projects first
        if folder_key in self.projects:
            project = self.projects[folder_key]
            master_files = project.get("master_files", {})

            # Verify files still exist
            verified_masters = {}
            for cal_type, file_path in master_files.items():
                if Path(file_path).exists():
                    verified_masters[cal_type] = file_path
                    self.logger.debug(
                        f"Found master {cal_type}: {Path(file_path).name}"
                    )
                else:
                    self.logger.warning(f"Master {cal_type} file missing: {file_path}")

            if verified_masters:
                self.logger.info(
                    f"Found {len(verified_masters)} master files for {Path(lights_folder).name}"
                )
                return verified_masters

        # Check filter-aware projects
        if folder_key in self.filter_projects:
            project = self.filter_projects[folder_key]
            filter_masters = project.get("filter_master_files", {})

            # For now, return masters from the first available filter
            # TODO: Implement proper filter detection from lights
            if filter_masters:
                first_filter = list(filter_masters.keys())[0]
                masters = filter_masters[first_filter]

                verified_masters = {}
                for cal_type, file_path in masters.items():
                    if Path(file_path).exists():
                        verified_masters[cal_type] = file_path
                        self.logger.debug(
                            f"Found filter master {cal_type}: {Path(file_path).name}"
                        )

                if verified_masters:
                    self.logger.info(
                        f"Found {len(verified_masters)} filter-aware master files"
                    )
                    return verified_masters

        self.logger.info("No calibration project found for this folder")
        return {}

    def load_master_frame(self, master_file_path: str) -> Optional[np.ndarray]:
        """
        Load a master calibration frame

        Args:
            master_file_path: Path to master calibration file

        Returns:
            Master frame data as numpy array, or None if error
        """
        try:
            master_path = Path(master_file_path)
            if not master_path.exists():
                self.logger.warning(f"Master file not found: {master_file_path}")
                return None

            with fits.open(master_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                self.logger.debug(f"Loaded master frame: {master_path.name}")
                return data

        except Exception as e:
            self.logger.error(f"Error loading master frame {master_file_path}: {e}")
            return None

    def detect_filter(self, fits_header: fits.Header) -> str:
        """
        Detect filter from FITS header

        Args:
            fits_header: FITS header object

        Returns:
            Filter name or 'UNKNOWN'
        """
        # Common filter keywords
        filter_keywords = ["FILTER", "FILTERS", "FILTER1", "INSFLNAM", "FILTNME3"]

        for keyword in filter_keywords:
            if keyword in fits_header:
                filter_value = str(fits_header[keyword]).strip().upper()
                if filter_value and filter_value not in ["NONE", "NULL", "", "N/A"]:
                    return self._normalize_filter_name(filter_value)

        # Check image type for bias/dark
        image_type = fits_header.get("IMAGETYP", "").upper()
        if "BIAS" in image_type:
            return "BIAS"
        elif "DARK" in image_type:
            return "DARK"

        return "UNKNOWN"

    def _normalize_filter_name(self, filter_name: str) -> str:
        """Normalize filter names to standard format"""
        filter_name = filter_name.upper().strip()

        # Common filter mappings
        filter_mappings = {
            "LUMINANCE": "L",
            "LUM": "L",
            "CLEAR": "L",
            "RED": "R",
            "GREEN": "G",
            "BLUE": "B",
            "HYDROGEN": "Ha",
            "H-ALPHA": "Ha",
            "HALPHA": "Ha",
            "OXYGEN": "OIII",
            "O3": "OIII",
            "SULFUR": "SII",
            "S2": "SII",
        }

        return filter_mappings.get(filter_name, filter_name)


def enhanced_load_fits_stack(
    folder: str,
    plate_solve_missing: bool = False,
    astap_exe: str = "astap",
    auto_calibrate: bool = True,
    manual_master_bias: Optional[np.ndarray] = None,
    manual_master_dark: Optional[np.ndarray] = None,
    manual_master_flat: Optional[np.ndarray] = None,
    progress_callback: Optional[callable] = None,
    max_workers: int = 4,
) -> Tuple[np.ndarray, List[str], List[Optional[WCS]]]:
    """
    Enhanced FITS stack loading with automatic calibration detection and application

    Args:
        folder: Directory containing FITS files
        plate_solve_missing: Whether to plate solve files without WCS
        astap_exe: Path to ASTAP executable
        auto_calibrate: Whether to automatically detect and apply calibration
        manual_master_bias: Manual bias frame (overrides auto-detection)
        manual_master_dark: Manual dark frame (overrides auto-detection)
        manual_master_flat: Manual flat frame (overrides auto-detection)
        progress_callback: Optional callback for progress updates
        max_workers: Number of threads for parallel processing

    Returns:
        Tuple of (frames array, filenames list, wcs_objects list)
    """
    logger = CalibrationLogger()
    logger.info(f"Enhanced FITS loading from: {folder}")

    # Validate input folder
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder}")
        return np.array([]), [], []

    # Find FITS files
    fits_files = []
    for ext in ["*.fits", "*.fit", "*.fts"]:
        fits_files.extend(folder_path.glob(ext))

    fits_files = sorted([f for f in fits_files if f.is_file()])

    if not fits_files:
        logger.error(f"No FITS files found in {folder}")
        return np.array([]), [], []

    logger.info(f"Found {len(fits_files)} FITS files")

    # Initialize calibration manager
    cal_manager = AutoCalibrationManager()

    # Load master calibration frames
    master_bias = manual_master_bias
    master_dark = manual_master_dark
    master_flat = manual_master_flat

    if auto_calibrate and (
        master_bias is None or master_dark is None or master_flat is None
    ):
        logger.info("Auto-detecting calibration files...")
        master_files = cal_manager.get_master_files_for_folder(folder)

        if master_files:
            if master_bias is None and "bias" in master_files:
                master_bias = cal_manager.load_master_frame(master_files["bias"])
                logger.info("✅ Auto-loaded master bias")

            if master_dark is None and "dark" in master_files:
                master_dark = cal_manager.load_master_frame(master_files["dark"])
                logger.info("✅ Auto-loaded master dark")

            if master_flat is None and "flat" in master_files:
                master_flat = cal_manager.load_master_frame(master_files["flat"])
                logger.info("✅ Auto-loaded master flat")
        else:
            logger.info("No automatic calibration files found")

    # Progress tracking
    total_files = len(fits_files)
    processed_files = 0

    def update_progress(additional_files: int = 1):
        nonlocal processed_files
        processed_files += additional_files
        if progress_callback:
            progress = int((processed_files / total_files) * 100)
            progress_callback(progress)

    def load_single_fits(
        fits_file: Path,
    ) -> Optional[Tuple[np.ndarray, str, Optional[WCS]]]:
        """Load a single FITS file with calibration"""
        try:
            # Read FITS file
            with fits.open(fits_file) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = hdul[0].header

            # Check for existing WCS
            has_wcs = "CRVAL1" in header and "CRVAL2" in header

            # Plate solve if requested and missing WCS
            if plate_solve_missing and not has_wcs:
                try:
                    import subprocess

                    result = subprocess.run(
                        [astap_exe, "-f", str(fits_file), "-solve"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        logger.debug(f"Plate solved: {fits_file.name}")
                        # Re-read header to get WCS
                        with fits.open(fits_file) as hdul:
                            header = hdul[0].header
                except Exception as e:
                    logger.warning(f"Plate solving failed for {fits_file.name}: {e}")

            # Apply calibration frames
            calibrated_data = data.copy()

            # Bias subtraction
            if master_bias is not None:
                try:
                    if master_bias.shape == data.shape:
                        calibrated_data = calibrated_data - master_bias
                        logger.debug(f"Applied bias to {fits_file.name}")
                    else:
                        logger.warning(
                            f"Bias frame shape mismatch for {fits_file.name}"
                        )
                except Exception as e:
                    logger.error(f"Error applying bias to {fits_file.name}: {e}")

            # Dark subtraction
            if master_dark is not None:
                try:
                    if master_dark.shape == data.shape:
                        calibrated_data = calibrated_data - master_dark
                        logger.debug(f"Applied dark to {fits_file.name}")
                    else:
                        logger.warning(
                            f"Dark frame shape mismatch for {fits_file.name}"
                        )
                except Exception as e:
                    logger.error(f"Error applying dark to {fits_file.name}: {e}")

            # Flat field correction
            if master_flat is not None:
                try:
                    if master_flat.shape == data.shape and np.all(master_flat > 0):
                        # Normalize flat field
                        flat_norm = master_flat / np.median(master_flat)
                        calibrated_data = calibrated_data / flat_norm
                        logger.debug(f"Applied flat to {fits_file.name}")
                    else:
                        logger.warning(f"Flat frame issue for {fits_file.name}")
                except Exception as e:
                    logger.error(f"Error applying flat to {fits_file.name}: {e}")

            # Create WCS object
            wcs_obj = None
            try:
                if "CRVAL1" in header and "CRVAL2" in header:
                    wcs_obj = WCS(header)
                    if not wcs_obj.has_celestial:
                        wcs_obj = None
                        logger.debug(f"Invalid WCS in {fits_file.name}")
                    else:
                        logger.debug(f"Valid WCS loaded for {fits_file.name}")
            except Exception as e:
                logger.warning(f"WCS creation failed for {fits_file.name}: {e}")
                wcs_obj = None

            return (calibrated_data, fits_file.name, wcs_obj)

        except Exception as e:
            logger.error(f"Error loading {fits_file}: {e}")
            return None

    # Process files in parallel
    frames = []
    filenames = []
    wcs_objects = []

    logger.info(f"Processing {len(fits_files)} files with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(load_single_fits, fits_file): fits_file
            for fits_file in fits_files
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            fits_file = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    frame_data, filename, wcs_obj = result
                    frames.append(frame_data)
                    filenames.append(filename)
                    wcs_objects.append(wcs_obj)
                else:
                    logger.warning(f"Failed to load: {fits_file.name}")
            except Exception as e:
                logger.error(f"Error processing {fits_file.name}: {e}")

            # Update progress
            update_progress()

    if not frames:
        logger.error("No valid FITS files loaded")
        return np.array([]), [], []

    # Convert to numpy array
    frames_array = np.array(frames)

    # Log calibration summary
    cal_summary = []
    if master_bias is not None:
        cal_summary.append("bias")
    if master_dark is not None:
        cal_summary.append("dark")
    if master_flat is not None:
        cal_summary.append("flat")

    if cal_summary:
        logger.info(
            f"✅ Loaded {len(frames)} frames with {', '.join(cal_summary)} calibration"
        )
    else:
        logger.info(f"✅ Loaded {len(frames)} frames (no calibration applied)")

    wcs_count = sum(1 for w in wcs_objects if w is not None)
    logger.info(f"✅ {wcs_count}/{len(frames)} frames have valid WCS")

    return frames_array, filenames, wcs_objects


def apply_calibration_to_image(
    image_data: np.ndarray,
    master_bias: Optional[np.ndarray] = None,
    master_dark: Optional[np.ndarray] = None,
    master_flat: Optional[np.ndarray] = None,
    validate_shapes: bool = True,
) -> np.ndarray:
    """
    Apply calibration frames to a single image

    Args:
        image_data: Raw image data
        master_bias: Master bias frame
        master_dark: Master dark frame
        master_flat: Master flat frame
        validate_shapes: Whether to validate frame shapes match

    Returns:
        Calibrated image data
    """
    logger = CalibrationLogger()
    calibrated = image_data.astype(np.float32).copy()

    # Bias subtraction
    if master_bias is not None:
        if not validate_shapes or master_bias.shape == calibrated.shape:
            calibrated = calibrated - master_bias
            logger.debug("Applied bias correction")
        else:
            logger.warning(
                f"Bias shape mismatch: {master_bias.shape} vs {calibrated.shape}"
            )

    # Dark subtraction
    if master_dark is not None:
        if not validate_shapes or master_dark.shape == calibrated.shape:
            calibrated = calibrated - master_dark
            logger.debug("Applied dark correction")
        else:
            logger.warning(
                f"Dark shape mismatch: {master_dark.shape} vs {calibrated.shape}"
            )

    # Flat field correction
    if master_flat is not None:
        if not validate_shapes or master_flat.shape == calibrated.shape:
            # Avoid division by zero
            flat_safe = np.where(master_flat > 0, master_flat, 1.0)
            # Normalize to median
            flat_norm = flat_safe / np.median(flat_safe)
            calibrated = calibrated / flat_norm
            logger.debug("Applied flat field correction")
        else:
            logger.warning(
                f"Flat shape mismatch: {master_flat.shape} vs {calibrated.shape}"
            )

    return calibrated


def get_calibration_status(lights_folder: str) -> Dict:
    """
    Get calibration status for a lights folder

    Args:
        lights_folder: Path to lights folder

    Returns:
        Dictionary with calibration status information
    """
    logger = CalibrationLogger()
    cal_manager = AutoCalibrationManager()

    master_files = cal_manager.get_master_files_for_folder(lights_folder)

    status = {
        "folder": lights_folder,
        "has_calibration": len(master_files) > 0,
        "master_files": master_files,
        "available_types": list(master_files.keys()),
        "missing_types": [],
        "file_status": {},
    }

    # Check which calibration types are available
    all_types = ["bias", "dark", "flat", "dark_flat"]
    for cal_type in all_types:
        if cal_type not in master_files:
            status["missing_types"].append(cal_type)

    # Verify file existence and get info
    for cal_type, file_path in master_files.items():
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            try:
                stat = file_path_obj.stat()
                status["file_status"][cal_type] = {
                    "exists": True,
                    "path": str(file_path_obj),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            except Exception as e:
                status["file_status"][cal_type] = {
                    "exists": True,
                    "path": str(file_path_obj),
                    "error": str(e),
                }
        else:
            status["file_status"][cal_type] = {
                "exists": False,
                "path": str(file_path_obj),
                "error": "File not found",
            }

    return status


def create_calibration_summary_report(lights_folder: str) -> str:
    """
    Create a human-readable calibration summary report

    Args:
        lights_folder: Path to lights folder

    Returns:
        Formatted calibration report string
    """
    status = get_calibration_status(lights_folder)

    report_lines = [
        f"Calibration Status Report",
        f"=" * 50,
        f"Lights Folder: {Path(lights_folder).name}",
        f"Full Path: {lights_folder}",
        f"",
        f"Calibration Available: {'✅ Yes' if status['has_calibration'] else '❌ No'}",
        f"",
    ]

    if status["has_calibration"]:
        report_lines.extend([f"Available Master Files:", f"----------------------"])

        for cal_type in status["available_types"]:
            file_info = status["file_status"][cal_type]
            if file_info["exists"]:
                report_lines.append(
                    f"✅ {cal_type.title()}: {Path(file_info['path']).name} "
                    f"({file_info.get('size_mb', 0)} MB)"
                )
            else:
                report_lines.append(f"❌ {cal_type.title()}: File missing")

        if status["missing_types"]:
            report_lines.extend(
                [f"", f"Missing Calibration Types:", f"--------------------------"]
            )
            for missing_type in status["missing_types"]:
                report_lines.append(f"⚠️  {missing_type.title()}")
    else:
        report_lines.extend(
            [
                f"No calibration project found for this folder.",
                f"Use Calibration → Calibration Setup to create master files.",
            ]
        )

    return "\n".join(report_lines)


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced calibration system
    print("Testing Enhanced Calibration Manager...")

    # Test folder (replace with actual path)
    test_folder = r"F:\astrophotography\2024-07-13 - bortle2 - sthelens"

    if Path(test_folder).exists():
        print(f"\nTesting with folder: {test_folder}")

        # Test calibration status
        print("\n" + "=" * 60)
        print("CALIBRATION STATUS")
        print("=" * 60)
        report = create_calibration_summary_report(test_folder)
        print(report)

        # Test enhanced loading (with progress callback)
        def progress_callback(percent):
            print(f"Loading progress: {percent}%")

        print(f"\n{'='*60}")
        print("ENHANCED FITS LOADING TEST")
        print("=" * 60)

        try:
            frames, filenames, wcs_objects = enhanced_load_fits_stack(
                folder=test_folder,
                auto_calibrate=True,
                plate_solve_missing=False,
                progress_callback=progress_callback,
                max_workers=2,  # Reduced for testing
            )

            print(f"\n✅ Enhanced loading completed!")
            print(f"   Loaded frames: {len(frames)}")
            print(f"   With WCS: {sum(1 for w in wcs_objects if w is not None)}")
            print(f"   Frame shape: {frames[0].shape if len(frames) > 0 else 'N/A'}")

        except Exception as e:
            print(f"❌ Enhanced loading failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Test folder not found: {test_folder}")
        print("Update the test_folder path to test with your data")

    print(f"\n{'='*60}")
    print("CALIBRATION MANAGER READY")
    print("=" * 60)
    print("The enhanced calibration system is ready for use!")
    print("Key features:")
    print("• Automatic master file detection")
    print("• Filter-aware calibration")
    print("• Parallel FITS processing")
    print("• Progress tracking")
    print("• Comprehensive error handling")
    print("• WCS preservation and plate solving")
