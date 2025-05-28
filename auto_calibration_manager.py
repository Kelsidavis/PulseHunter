"""
Auto Calibration Manager for PulseHunter
Handles automatic detection and loading of master calibration files
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.io import fits


class AutoCalibrationManager:
    """Manages automatic calibration file detection and usage"""

    def __init__(self):
        self.config_file = Path("calibration_projects.json")
        self.projects = self.load_projects()

    def load_projects(self):
        """Load calibration projects configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def get_calibration_for_folder(self, lights_folder: str) -> Dict[str, str]:
        """
        Get calibration files for a specific lights folder

        Args:
            lights_folder: Path to folder containing light frames

        Returns:
            Dictionary of calibration type -> file path
        """
        project_id = str(Path(lights_folder).resolve())

        if project_id in self.projects:
            project = self.projects[project_id]
            master_files = project.get("master_files", {})

            # Verify files still exist
            valid_masters = {}
            for cal_type, file_path in master_files.items():
                if Path(file_path).exists():
                    valid_masters[cal_type] = file_path
                else:
                    print(f"Warning: Master {cal_type} file missing: {file_path}")

            return valid_masters

        return {}

    def load_master_frame(self, file_path: str) -> Optional[np.ndarray]:
        """Load a master calibration frame"""
        try:
            if not Path(file_path).exists():
                print(f"Master file not found: {file_path}")
                return None

            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                print(f"✅ Loaded master file: {Path(file_path).name}")
                return data

        except Exception as e:
            print(f"Error loading master file {file_path}: {e}")
            return None


def enhanced_load_fits_stack(
    folder,
    plate_solve_missing=False,
    astap_exe="astap",
    auto_calibrate=True,  # NEW: Enable automatic calibration
    manual_master_bias=None,
    manual_master_dark=None,
    manual_master_flat=None,
    camera_mode="mono",
    filter_name=None,
):
    """
    Enhanced FITS loading with automatic calibration detection

    This is a drop-in replacement for the original load_fits_stack function
    that automatically detects and applies calibration files.

    Args:
        folder: Directory containing FITS files
        plate_solve_missing: Whether to plate solve files without WCS
        astap_exe: Path to ASTAP executable
        auto_calibrate: Whether to automatically detect and apply calibration
        manual_master_bias: Manually specified master bias frame
        manual_master_dark: Manually specified master dark frame
        manual_master_flat: Manually specified master flat frame
        camera_mode: Camera mode ("mono" or "osc")
        filter_name: Filter name for the observation

    Returns:
        tuple: (frames array, filenames list, wcs_objects list)
    """
    import pulsehunter_core  # Import original module

    # Determine calibration files to use
    master_bias = manual_master_bias
    master_dark = manual_master_dark
    master_flat = manual_master_flat

    if auto_calibrate:
        print("🔍 Checking for automatic calibration files...")

        # Get automatic calibration files
        cal_manager = AutoCalibrationManager()
        auto_masters = cal_manager.get_calibration_for_folder(folder)

        if auto_masters:
            print(
                f"✅ Found automatic calibration configuration for: {Path(folder).name}"
            )

            # Load master frames (only if not manually specified)
            if not master_bias and "bias" in auto_masters:
                master_bias = cal_manager.load_master_frame(auto_masters["bias"])

            if not master_dark and "dark" in auto_masters:
                master_dark = cal_manager.load_master_frame(auto_masters["dark"])

            if not master_flat and "flat" in auto_masters:
                master_flat = cal_manager.load_master_frame(auto_masters["flat"])

            print("🎯 Automatic calibration files loaded and ready!")
        else:
            print("ℹ️  No automatic calibration found for this folder")

    # Call original function with calibration files
    return pulsehunter_core.load_fits_stack(
        folder=folder,
        plate_solve_missing=plate_solve_missing,
        astap_exe=astap_exe,
        master_bias=master_bias,
        master_dark=master_dark,
        master_flat=master_flat,
        camera_mode=camera_mode,
        filter_name=filter_name,
    )


def get_master_files_for_folder(lights_folder):
    """
    Get master calibration files for a specific lights folder
    This function should be called from the main processing code
    """
    cal_manager = AutoCalibrationManager()
    return cal_manager.get_calibration_for_folder(lights_folder)


if __name__ == "__main__":
    # Test the auto calibration manager
    cal_manager = AutoCalibrationManager()
    print(f"Calibration projects loaded: {len(cal_manager.projects)}")

    for project_id, project in cal_manager.projects.items():
        print(f"Project: {Path(project['lights_folder']).name}")
        print(f"  Master files: {list(project.get('master_files', {}).keys())}")
