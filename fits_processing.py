"""
FITS File Processing Utilities for PulseHunter
Handles actual astronomical image processing tasks
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports - will fall back to simulation if not available
try:
    from astropy.io import fits
    from astropy.stats import sigma_clip, sigma_clipped_stats
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available - using simulation mode for FITS processing")

from calibration_utilities import CalibrationConfig, CalibrationLogger


class FITSProcessor:
    """Handle FITS file reading, validation, and processing"""

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.logger = CalibrationLogger()

    def read_fits_file(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """
        Read a FITS file and extract header information and data

        Args:
            file_path: Path to FITS file

        Returns:
            Dictionary with header info and data, or None if error
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"FITS file not found: {file_path}")
            return None

        try:
            if ASTROPY_AVAILABLE:
                return self._read_fits_with_astropy(file_path)
            else:
                return self._simulate_fits_read(file_path)
        except Exception as e:
            self.logger.error(f"Error reading FITS file {file_path}: {e}")
            return None

    def _read_fits_with_astropy(self, file_path: Path) -> Dict:
        """Read FITS file using astropy"""
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            data = hdul[0].data

            # Extract key information
            fits_info = {
                "file_path": str(file_path),
                "data": data,
                "header": dict(header),
                "exposure_time": header.get("EXPTIME", 0.0),
                "image_type": header.get("IMAGETYP", "UNKNOWN"),
                "ccd_temp": header.get("CCD-TEMP", -999),
                "binning": f"{header.get('XBINNING', 1)}x{header.get('YBINNING', 1)}",
                "date_obs": header.get("DATE-OBS", ""),
                "object": header.get("OBJECT", ""),
                "filter": header.get("FILTER", ""),
                "dimensions": data.shape if data is not None else (0, 0),
                "bitpix": header.get("BITPIX", 16),
                "bzero": header.get("BZERO", 0),
                "bscale": header.get("BSCALE", 1),
            }

            # Calculate basic statistics
            if data is not None:
                fits_info.update(self._calculate_statistics(data))

            return fits_info

    def _simulate_fits_read(self, file_path: Path) -> Dict:
        """Simulate FITS file reading for testing without astropy"""
        import random

        # Create simulated data
        width, height = 1024, 1024
        simulated_data = np.random.normal(1000, 50, (height, width)).astype(np.uint16)

        # Simulate header information
        fits_info = {
            "file_path": str(file_path),
            "data": simulated_data,
            "header": {
                "EXPTIME": 300.0 if "dark" in file_path.name.lower() else 1.0,
                "IMAGETYP": self._guess_image_type(file_path.name),
                "CCD-TEMP": -20.0 + random.uniform(-2, 2),
                "XBINNING": 1,
                "YBINNING": 1,
                "DATE-OBS": "2024-01-01T00:00:00",
                "OBJECT": "Calibration",
                "FILTER": "None",
                "BITPIX": 16,
                "BZERO": 32768,
                "BSCALE": 1,
            },
            "exposure_time": 300.0 if "dark" in file_path.name.lower() else 1.0,
            "image_type": self._guess_image_type(file_path.name),
            "ccd_temp": -20.0 + random.uniform(-2, 2),
            "binning": "1x1",
            "date_obs": "2024-01-01T00:00:00",
            "object": "Calibration",
            "filter": "None",
            "dimensions": (height, width),
            "bitpix": 16,
            "bzero": 32768,
            "bscale": 1,
        }

        # Calculate statistics
        fits_info.update(self._calculate_statistics(simulated_data))

        return fits_info

    def _guess_image_type(self, filename: str) -> str:
        """Guess image type from filename"""
        filename_lower = filename.lower()
        if "bias" in filename_lower:
            return "BIAS"
        elif "dark" in filename_lower:
            return "DARK"
        elif "flat" in filename_lower:
            return "FLAT"
        else:
            return "LIGHT"

    def _calculate_statistics(self, data: np.ndarray) -> Dict:
        """Calculate basic image statistics"""
        try:
            if ASTROPY_AVAILABLE:
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            else:
                mean = np.mean(data)
                median = np.median(data)
                std = np.std(data)

            return {
                "mean": float(mean),
                "median": float(median),
                "std": float(std),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "data_type": str(data.dtype),
            }
        except Exception as e:
            self.logger.warning(f"Error calculating statistics: {e}")
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "data_type": str(data.dtype) if hasattr(data, "dtype") else "unknown",
            }


class CalibrationProcessor:
    """Process calibration frames to create master calibration files"""

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.logger = CalibrationLogger()
        self.fits_processor = FITSProcessor(config)

    def create_master_calibration(
        self,
        input_files: List[Path],
        output_file: Path,
        calibration_type: str,
        progress_callback=None,
    ) -> bool:
        """
        Create master calibration file from input frames

        Args:
            input_files: List of input FITS files
            output_file: Output path for master file
            calibration_type: Type of calibration (bias, dark, flat, dark_flat)
            progress_callback: Optional callback for progress updates

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(
            f"Creating master {calibration_type} from {len(input_files)} files"
        )

        try:
            # Read all input files
            frames_data = []
            valid_files = []

            total_files = len(input_files)
            for i, file_path in enumerate(input_files):
                # Progress per-frame, up to 90% of total
                if progress_callback:
                    percent = int((i / max(total_files, 1)) * 90)
                    progress_callback(percent)
                fits_info = self.fits_processor.read_fits_file(file_path)
                if fits_info and fits_info["data"] is not None:
                    frames_data.append(fits_info["data"])
                    valid_files.append(file_path)
                    self.logger.debug(f"Loaded: {file_path.name}")

            if len(frames_data) < 1:
                self.logger.error("No valid input files found for calibration")
                if progress_callback:
                    progress_callback(100)
                return False

            # Stack/Combine
            if calibration_type in ["bias", "dark", "flat", "dark_flat"]:
                combined = np.median(frames_data, axis=0)
            else:
                combined = np.median(frames_data, axis=0)
            # After stacking, advance to 95%
            if progress_callback:
                progress_callback(95)

            # Save to output file
            primary_header = fits.getheader(str(valid_files[0]))
            now = datetime.now().isoformat()
            primary_header.add_history(
                f"PulseHunter master {calibration_type} created from {len(valid_files)} frames on {now}"
            )

            fits.writeto(output_file, combined, primary_header, overwrite=True)
            if progress_callback:
                progress_callback(100)

            self.logger.info(f"✅ Master {calibration_type} saved: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating master {calibration_type}: {e}")
            if progress_callback:
                progress_callback(100)
            return False


class ImageAnalyzer:
    """Analyze processed images for quality and characteristics"""

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.logger = CalibrationLogger()

    def analyze_master_frame(self, file_path: Path) -> Dict:
        """Analyze a master calibration frame"""
        fits_processor = FITSProcessor(self.config)
        fits_info = fits_processor.read_fits_file(file_path)

        if not fits_info:
            return {"error": "Could not read file"}

        analysis = {
            "file_info": {
                "path": str(file_path),
                "size_mb": file_path.stat().st_size / 1024 / 1024,
                "created": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat(),
            },
            "image_info": {
                "dimensions": fits_info["dimensions"],
                "data_type": fits_info.get("data_type", "unknown"),
                "image_type": fits_info.get("image_type", "unknown"),
            },
            "statistics": {
                "mean": fits_info.get("mean", 0),
                "median": fits_info.get("median", 0),
                "std": fits_info.get("std", 0),
                "min": fits_info.get("min", 0),
                "max": fits_info.get("max", 0),
            },
        }

        # Add quality assessment
        analysis["quality"] = self._assess_quality(fits_info)

        return analysis

    def _assess_quality(self, fits_info: Dict) -> Dict:
        """Assess the quality of a calibration frame"""
        quality = {"overall": "unknown", "issues": [], "score": 0.0}

        try:
            data = fits_info.get("data")
            if data is None:
                quality["issues"].append("No image data")
                return quality

            stats = fits_info
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)

            score = 100.0  # Start with perfect score

            # Check for saturation
            if hasattr(data, "dtype"):
                if data.dtype == np.uint16:
                    max_possible = 65535
                elif data.dtype == np.uint8:
                    max_possible = 255
                else:
                    max_possible = max_val * 1.1

                saturation_ratio = np.sum(data >= max_possible * 0.95) / data.size
                if saturation_ratio > 0.01:  # More than 1% saturated
                    quality["issues"].append(f"High saturation: {saturation_ratio:.1%}")
                    score -= 20

            # Check for reasonable noise levels
            if mean > 0:
                snr = mean / std if std > 0 else 0
                if snr < 5:
                    quality["issues"].append(f"Low SNR: {snr:.1f}")
                    score -= 15
                elif snr > 1000:
                    quality["issues"].append(f"Unusually high SNR: {snr:.1f}")
                    score -= 5

            # Check for dead pixels (very low values)
            if data.size > 0:
                dead_pixel_ratio = np.sum(data <= min_val * 1.1) / data.size
                if dead_pixel_ratio > 0.05:  # More than 5% dead pixels
                    quality["issues"].append(
                        f"Many dead pixels: {dead_pixel_ratio:.1%}"
                    )
                    score -= 10

            quality["score"] = max(0, score)

            # Overall assessment
            if score >= 90:
                quality["overall"] = "excellent"
            elif score >= 75:
                quality["overall"] = "good"
            elif score >= 50:
                quality["overall"] = "fair"
            else:
                quality["overall"] = "poor"

        except Exception as e:
            quality["issues"].append(f"Analysis error: {e}")

        return quality


# Convenience functions
def create_master_bias(
    input_folder: Path, output_file: Path, progress_callback=None
) -> bool:
    """Create master bias frame"""
    processor = CalibrationProcessor()
    bias_files = list(input_folder.glob("*bias*.fit*")) + list(
        input_folder.glob("*BIAS*.fit*")
    )
    return processor.create_master_calibration(
        bias_files, output_file, "bias", progress_callback
    )


def create_master_dark(
    input_folder: Path, output_file: Path, progress_callback=None
) -> bool:
    """Create master dark frame"""
    processor = CalibrationProcessor()
    dark_files = list(input_folder.glob("*dark*.fit*")) + list(
        input_folder.glob("*DARK*.fit*")
    )
    return processor.create_master_calibration(
        dark_files, output_file, "dark", progress_callback
    )


def create_master_flat(
    input_folder: Path, output_file: Path, progress_callback=None
) -> bool:
    """Create master flat frame"""
    processor = CalibrationProcessor()
    flat_files = list(input_folder.glob("*flat*.fit*")) + list(
        input_folder.glob("*FLAT*.fit*")
    )
    return processor.create_master_calibration(
        flat_files, output_file, "flat", progress_callback
    )


def analyze_master_file(file_path: Path) -> Dict:
    """Analyze a master calibration file"""
    analyzer = ImageAnalyzer()
    return analyzer.analyze_master_frame(file_path)


# Example usage
if __name__ == "__main__":
    # Test the FITS processing system
    print("Testing FITS processing system...")

    config = CalibrationConfig()
    processor = FITSProcessor(config)

    # Test with a simulated file
    test_file = Path("test_bias_001.fits")
    fits_info = processor.read_fits_file(test_file)

    if fits_info:
        print(f"Successfully processed: {test_file}")
        print(f"Dimensions: {fits_info['dimensions']}")
        print(f"Mean: {fits_info['mean']:.2f}")
        print(f"Image type: {fits_info['image_type']}")
    else:
        print("Failed to process test file")

    print("\nFITS processing system ready!")
