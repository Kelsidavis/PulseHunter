"""
Simplified Calibration Manager for PulseHunter
Handles bias, dark, and flat frame calibration
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger(__name__)


class CalibrationProject:
    """Manages calibration projects and master files"""

    def __init__(self, project_file: str = "calibration_projects.json"):
        self.project_file = Path(project_file)
        self.projects = self._load_projects()

    def _load_projects(self) -> Dict:
        """Load existing calibration projects"""
        if self.project_file.exists():
            try:
                with open(self.project_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load projects: {e}")
        return {}

    def save_projects(self):
        """Save calibration projects"""
        try:
            with open(self.project_file, "w") as f:
                json.dump(self.projects, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save projects: {e}")

    def create_project(self, lights_folder: str, master_files: Dict[str, str]) -> str:
        """Create a new calibration project"""
        project_id = str(Path(lights_folder).resolve())
        self.projects[project_id] = {
            "lights_folder": lights_folder,
            "master_files": master_files,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        self.save_projects()
        logger.info(f"Created calibration project for {lights_folder}")
        return project_id

    def get_project(self, lights_folder: str) -> Optional[Dict]:
        """Get calibration project for a folder"""
        project_id = str(Path(lights_folder).resolve())
        if project_id in self.projects:
            self.projects[project_id]["last_used"] = datetime.now().isoformat()
            self.save_projects()
            return self.projects[project_id]
        return None


class CalibrationManager:
    """Main calibration manager for creating and applying calibrations"""

    def __init__(self):
        self.project_manager = CalibrationProject()
        self.logger = logger

    def create_master_bias(
        self,
        bias_files: List[Path],
        output_file: Path,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Create master bias frame from multiple bias frames"""
        return self._create_master_calibration(
            bias_files, output_file, "bias", progress_callback
        )

    def create_master_dark(
        self,
        dark_files: List[Path],
        output_file: Path,
        master_bias: Optional[np.ndarray] = None,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Create master dark frame from multiple dark frames"""
        return self._create_master_calibration(
            dark_files, output_file, "dark", progress_callback, master_bias
        )

    def create_master_flat(
        self,
        flat_files: List[Path],
        output_file: Path,
        master_bias: Optional[np.ndarray] = None,
        master_dark: Optional[np.ndarray] = None,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Create master flat frame from multiple flat frames"""
        return self._create_master_calibration(
            flat_files, output_file, "flat", progress_callback, master_bias, master_dark
        )

    def _create_master_calibration(
        self,
        input_files: List[Path],
        output_file: Path,
        cal_type: str,
        progress_callback: Optional[callable] = None,
        master_bias: Optional[np.ndarray] = None,
        master_dark: Optional[np.ndarray] = None,
    ) -> bool:
        """Generic method to create master calibration frames"""

        if not input_files:
            self.logger.error(f"No {cal_type} files provided")
            return False

        self.logger.info(f"Creating master {cal_type} from {len(input_files)} files")

        try:
            # Load all frames
            frames = []
            for i, file_path in enumerate(input_files):
                try:
                    with fits.open(file_path) as hdul:
                        data = hdul[0].data.astype(np.float32)

                        # Apply calibrations if provided
                        if master_bias is not None:
                            data = data - master_bias
                        if master_dark is not None and cal_type == "flat":
                            data = data - master_dark

                        frames.append(data)

                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
                    continue

                if progress_callback:
                    progress = int((i + 1) / len(input_files) * 70)
                    progress_callback(progress)

            if len(frames) < 3:
                self.logger.error(f"Need at least 3 valid {cal_type} frames")
                return False

            # Stack frames using median
            self.logger.info(f"Stacking {len(frames)} {cal_type} frames...")
            master = np.median(frames, axis=0)

            if progress_callback:
                progress_callback(85)

            # For flats, normalize
            if cal_type == "flat":
                median_value = np.median(master[master > 0])
                if median_value > 0:
                    master = master / median_value
                else:
                    self.logger.error("Failed to normalize flat - median is zero")
                    return False

            # Save master frame
            header = fits.Header()
            header["IMAGETYP"] = f"MASTER_{cal_type.upper()}"
            header["NFRAMES"] = len(frames)
            header["DATE"] = datetime.utcnow().isoformat()
            header["COMMENT"] = f"PulseHunter master {cal_type} frame"

            # Add statistics
            mean, median, std = sigma_clipped_stats(master, sigma=3.0)
            header["MEAN"] = float(mean)
            header["MEDIAN"] = float(median)
            header["STDDEV"] = float(std)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            fits.writeto(output_file, master, header, overwrite=True)

            if progress_callback:
                progress_callback(100)

            self.logger.info(f"Saved master {cal_type} to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create master {cal_type}: {e}")
            return False

    def load_master_frame(self, file_path: Path) -> Optional[np.ndarray]:
        """Load a master calibration frame"""
        try:
            with fits.open(file_path) as hdul:
                return hdul[0].data.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Failed to load master frame {file_path}: {e}")
            return None

    def calibrate_lights(
        self,
        lights_folder: Path,
        output_folder: Path,
        master_bias: Optional[Path] = None,
        master_dark: Optional[Path] = None,
        master_flat: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[int, int]:
        """
        Calibrate light frames and save to output folder

        Returns:
            Tuple of (successful_count, total_count)
        """
        # Load master frames
        bias_data = self.load_master_frame(master_bias) if master_bias else None
        dark_data = self.load_master_frame(master_dark) if master_dark else None
        flat_data = self.load_master_frame(master_flat) if master_flat else None

        # Find light frames
        light_files = []
        for ext in ["*.fits", "*.fit", "*.fts", "*.FIT", "*.FITS"]:
            light_files.extend(lights_folder.glob(ext))

        if not light_files:
            self.logger.error(f"No light frames found in {lights_folder}")
            return 0, 0

        self.logger.info(f"Calibrating {len(light_files)} light frames...")

        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)

        # Process files
        successful = 0
        total = len(light_files)

        for i, light_file in enumerate(light_files):
            try:
                # Load light frame
                with fits.open(light_file) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    header = hdul[0].header.copy()

                # Apply calibrations
                calibrated = data.copy()

                if bias_data is not None and bias_data.shape == data.shape:
                    calibrated = calibrated - bias_data
                    header.add_history("Bias correction applied")

                if dark_data is not None and dark_data.shape == data.shape:
                    calibrated = calibrated - dark_data
                    header.add_history("Dark correction applied")

                if flat_data is not None and flat_data.shape == data.shape:
                    # Avoid division by zero
                    flat_safe = np.where(flat_data > 0.1, flat_data, 1.0)
                    calibrated = calibrated / flat_safe
                    header.add_history("Flat field correction applied")

                # Add calibration info to header
                header["CALIBRAT"] = True
                header["CAL-DATE"] = datetime.utcnow().isoformat()
                if master_bias:
                    header["CAL-BIAS"] = master_bias.name
                if master_dark:
                    header["CAL-DARK"] = master_dark.name
                if master_flat:
                    header["CAL-FLAT"] = master_flat.name

                # Save calibrated frame
                output_file = output_folder / f"cal_{light_file.name}"
                fits.writeto(output_file, calibrated, header, overwrite=True)

                successful += 1

            except Exception as e:
                self.logger.error(f"Failed to calibrate {light_file}: {e}")

            if progress_callback:
                progress = int((i + 1) / total * 100)
                progress_callback(progress)

        self.logger.info(f"Calibrated {successful}/{total} light frames")
        return successful, total

    def get_calibration_for_folder(
        self, lights_folder: str
    ) -> Optional[Dict[str, str]]:
        """Get master calibration files for a specific lights folder"""
        project = self.project_manager.get_project(lights_folder)
        if project:
            master_files = project.get("master_files", {})
            # Verify files still exist
            verified = {}
            for cal_type, file_path in master_files.items():
                if Path(file_path).exists():
                    verified[cal_type] = file_path
            return verified if verified else None
        return None

    def register_calibration_project(
        self,
        lights_folder: str,
        master_bias: Optional[str] = None,
        master_dark: Optional[str] = None,
        master_flat: Optional[str] = None,
    ) -> str:
        """Register a calibration project for a lights folder"""
        master_files = {}
        if master_bias and Path(master_bias).exists():
            master_files["bias"] = master_bias
        if master_dark and Path(master_dark).exists():
            master_files["dark"] = master_dark
        if master_flat and Path(master_flat).exists():
            master_files["flat"] = master_flat

        return self.project_manager.create_project(lights_folder, master_files)


def quick_calibrate_folder(
    lights_folder: str,
    bias_folder: Optional[str] = None,
    dark_folder: Optional[str] = None,
    flat_folder: Optional[str] = None,
    output_folder: Optional[str] = None,
) -> bool:
    """
    Quick calibration helper function

    Args:
        lights_folder: Folder containing light frames
        bias_folder: Folder containing bias frames
        dark_folder: Folder containing dark frames
        flat_folder: Folder containing flat frames
        output_folder: Where to save calibrated lights (default: lights_folder/../calibrated)

    Returns:
        True if successful
    """
    manager = CalibrationManager()

    lights_path = Path(lights_folder)
    if not lights_path.exists():
        logger.error(f"Lights folder not found: {lights_folder}")
        return False

    # Set up paths
    if output_folder:
        output_path = Path(output_folder)
    else:
        output_path = lights_path.parent / "calibrated"

    master_path = lights_path.parent / "masters"
    master_path.mkdir(exist_ok=True)

    # Create master frames
    master_bias_file = None
    master_dark_file = None
    master_flat_file = None

    try:
        # Create master bias
        if bias_folder and Path(bias_folder).exists():
            bias_files = list(Path(bias_folder).glob("*.fit*"))
            if bias_files:
                master_bias_file = master_path / "master_bias.fits"
                logger.info(f"Creating master bias from {len(bias_files)} files...")
                if not manager.create_master_bias(bias_files, master_bias_file):
                    logger.warning("Failed to create master bias")
                    master_bias_file = None

        # Load master bias for dark/flat processing
        master_bias_data = None
        if master_bias_file and master_bias_file.exists():
            master_bias_data = manager.load_master_frame(master_bias_file)

        # Create master dark
        if dark_folder and Path(dark_folder).exists():
            dark_files = list(Path(dark_folder).glob("*.fit*"))
            if dark_files:
                master_dark_file = master_path / "master_dark.fits"
                logger.info(f"Creating master dark from {len(dark_files)} files...")
                if not manager.create_master_dark(
                    dark_files, master_dark_file, master_bias_data
                ):
                    logger.warning("Failed to create master dark")
                    master_dark_file = None

        # Load master dark for flat processing
        master_dark_data = None
        if master_dark_file and master_dark_file.exists():
            master_dark_data = manager.load_master_frame(master_dark_file)

        # Create master flat
        if flat_folder and Path(flat_folder).exists():
            flat_files = list(Path(flat_folder).glob("*.fit*"))
            if flat_files:
                master_flat_file = master_path / "master_flat.fits"
                logger.info(f"Creating master flat from {len(flat_files)} files...")
                if not manager.create_master_flat(
                    flat_files, master_flat_file, master_bias_data, master_dark_data
                ):
                    logger.warning("Failed to create master flat")
                    master_flat_file = None

        # Register calibration project
        manager.register_calibration_project(
            lights_folder,
            str(master_bias_file) if master_bias_file else None,
            str(master_dark_file) if master_dark_file else None,
            str(master_flat_file) if master_flat_file else None,
        )

        # Calibrate lights
        logger.info(f"Calibrating light frames...")
        successful, total = manager.calibrate_lights(
            lights_path,
            output_path,
            master_bias_file,
            master_dark_file,
            master_flat_file,
        )

        logger.info(f"Calibration complete: {successful}/{total} frames processed")
        logger.info(f"Calibrated files saved to: {output_path}")

        return successful > 0

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return False
