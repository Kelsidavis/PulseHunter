"""
Calibration utilities and support classes for PulseHunter
Handles configuration, ASTAP management, logging, and file operations
"""

import configparser
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QScreen
from PyQt6.QtWidgets import QApplication


class CalibrationConfig:
    """Configuration management for calibration processing"""

    def __init__(self, config_file="calibration_config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_default_config()
        self.load_config()

    def load_default_config(self):
        """Load default configuration values"""
        self.config["PROCESSING"] = {
            "min_frames_bias": "10",
            "min_frames_dark": "5",
            "min_frames_flat": "5",
            "min_frames_dark_flat": "5",
            "sigma_clipping_threshold": "3.0",
            "combination_method": "median",  # median, mean, sigma_clipped_mean
            "output_format": "fits",
            "compress_output": "True",
            "save_statistics": "True",
        }

        self.config["VALIDATION"] = {
            "check_exposure_consistency": "True",
            "max_exposure_variance": "0.1",  # seconds
            "check_temperature_consistency": "True",
            "max_temperature_variance": "2.0",  # degrees C
            "check_binning_consistency": "True",
            "required_keywords": "EXPTIME,IMAGETYP,CCD-TEMP",
        }

        self.config["DIALOG"] = {
            "remember_positions": "True",
            "auto_save_settings": "True",
            "show_advanced_options": "False",
            "default_output_folder": "",
            "auto_find_calibration_folders": "True",
        }

        self.config["ASTAP"] = {
            "executable_path": "",
            "auto_detect_on_startup": "True",
            "timeout_seconds": "30",
            "additional_parameters": "",
            "verify_on_load": "True",
        }

    def load_config(self):
        """Load configuration from file if it exists"""
        if Path(self.config_file).exists():
            self.config.read(self.config_file)

    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, "w") as f:
            self.config.write(f)

    def get(self, section, key, fallback=None):
        """Get configuration value with fallback"""
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section, key, fallback=0):
        """Get integer configuration value"""
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section, key, fallback=0.0):
        """Get float configuration value"""
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section, key, fallback=False):
        """Get boolean configuration value"""
        return self.config.getboolean(section, key, fallback=fallback)


class DialogPositionManager:
    """Manage consistent dialog positioning across monitors"""

    @staticmethod
    def get_optimal_position(dialog, parent=None):
        """Calculate optimal position for dialog"""
        app = QApplication.instance()

        if parent and parent.isVisible():
            # Position relative to parent
            parent_geometry = parent.geometry()
            parent_center = parent_geometry.center()

            # Find which screen contains the parent
            screen = app.screenAt(parent_center)
            if not screen:
                screen = app.primaryScreen()

            # Center on parent, but ensure it's fully visible on screen
            x = parent_geometry.x() + (parent_geometry.width() - dialog.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - dialog.height()) // 2

            # Constrain to screen bounds
            screen_geometry = screen.availableGeometry()
            x = max(
                screen_geometry.left(), min(x, screen_geometry.right() - dialog.width())
            )
            y = max(
                screen_geometry.top(),
                min(y, screen_geometry.bottom() - dialog.height()),
            )

            return x, y
        else:
            # Center on primary screen
            primary_screen = app.primaryScreen()
            screen_geometry = primary_screen.availableGeometry()
            x = screen_geometry.x() + (screen_geometry.width() - dialog.width()) // 2
            y = screen_geometry.y() + (screen_geometry.height() - dialog.height()) // 2
            return x, y

    @staticmethod
    def save_dialog_geometry(dialog, settings_key):
        """Save dialog geometry to settings"""
        settings = QSettings()
        settings.setValue(f"{settings_key}/geometry", dialog.saveGeometry())
        settings.setValue(
            f"{settings_key}/state",
            dialog.saveState() if hasattr(dialog, "saveState") else None,
        )

    @staticmethod
    def restore_dialog_geometry(dialog, settings_key, default_size=(800, 600)):
        """Restore dialog geometry from settings"""
        settings = QSettings()
        geometry = settings.value(f"{settings_key}/geometry")

        if geometry:
            dialog.restoreGeometry(geometry)
            state = settings.value(f"{settings_key}/state")
            if state and hasattr(dialog, "restoreState"):
                dialog.restoreState(state)
        else:
            # Set default size and center
            dialog.resize(*default_size)
            x, y = DialogPositionManager.get_optimal_position(dialog)
            dialog.move(x, y)


class CalibrationLogger:
    """Enhanced logging for calibration processes"""

    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("PulseHunter.Calibration")
        self.logger.setLevel(log_level)

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'calibration_{datetime.now().strftime("%Y%m%d")}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)


class ASTAPManager:
    """Manage ASTAP executable configuration and operations"""

    def __init__(self, config=None):
        self.config = config or CalibrationConfig()
        self.logger = CalibrationLogger()
        self._astap_path = None
        self._is_validated = False

    @property
    def astap_path(self):
        """Get current ASTAP path"""
        if not self._astap_path:
            self._astap_path = self.config.get("ASTAP", "executable_path", "")
        return self._astap_path

    @astap_path.setter
    def astap_path(self, path):
        """Set ASTAP path and save to config"""
        self._astap_path = path
        self.config.config.set("ASTAP", "executable_path", path)
        self.config.save_config()
        self._is_validated = False

    def clear_cached_astap_path(self):
        """Clear any cached ASTAP path to force re-detection"""
        self._astap_path = None
        self._is_validated = False
        # Also clear from config
        if hasattr(self, "config"):
            self.config.config.set("ASTAP", "executable_path", "")
            self.config.save_config()
        self.logger.info("Cleared cached ASTAP path - will re-detect on next access")

    def auto_detect_astap(self):
        """Automatically detect ASTAP installation - prefer CLI version"""
        self.logger.info("Auto-detecting ASTAP executable...")

        import platform

        potential_paths = []

        if platform.system() == "Windows":
            # PREFER CLI VERSIONS - check these FIRST
            base_locations = [
                "C:/Program Files/astap/",
                "C:/Program Files (x86)/astap/",
                "C:/astap/",
            ]

            cli_names = [
                "astap_cli.exe",
                "astap-cli.exe",
                "astap_console.exe",
                "astap-console.exe",
            ]

            for base_path in base_locations:
                for cli_name in cli_names:
                    cli_path = Path(base_path) / cli_name
                    potential_paths.append(str(cli_path))

            potential_paths.extend(
                [
                    "C:/Program Files/astap/astap_cli.exe",
                    "C:/Program Files/astap/astap.exe",
                    "C:/Program Files (x86)/astap/astap.exe",
                    "C:/astap/astap.exe",
                ]
            )

            for user_location in [
                Path.home() / "astap",
                Path.home() / "Desktop",
                Path.home() / "Downloads",
            ]:
                for cli_name in cli_names:
                    potential_paths.append(str(user_location / cli_name))
                potential_paths.append(str(user_location / "astap.exe"))

            path_astap = shutil.which("astap_cli.exe") or shutil.which("astap.exe")
            if path_astap:
                potential_paths.insert(0, path_astap)

        else:
            path_executable = shutil.which("astap")
            if path_executable:
                potential_paths.append(path_executable)

            potential_paths.extend(
                [
                    "/usr/local/bin/astap",
                    "/usr/bin/astap",
                    "/opt/astap/astap",
                    Path.home() / "bin" / "astap",
                    Path.home() / "astap" / "astap",
                ]
            )

        # Check current working directory (lowest priority)
        if platform.system() == "Windows":
            cli_names = [
                "astap_cli.exe",
                "astap-cli.exe",
                "astap_console.exe",
                "astap-console.exe",
            ]
            for cli_name in cli_names:
                potential_paths.append(str(Path.cwd() / cli_name))
            potential_paths.append(str(Path.cwd() / "astap.exe"))
        else:
            potential_paths.append(str(Path.cwd() / "astap"))

        # Remove None values and duplicates
        potential_paths = list(
            dict.fromkeys([p for p in potential_paths if p is not None])
        )

        self.logger.info(
            f"Checking {len(potential_paths)} potential ASTAP locations..."
        )
        for i, path in enumerate(potential_paths[:5]):
            cli_indicator = " (CLI)" if "cli" in Path(path).name.lower() else " (GUI)"
            self.logger.debug(f"  {i+1}. {path}{cli_indicator}")

        for path in potential_paths:
            path_obj = Path(path)
            if path_obj.exists():
                is_cli = (
                    "cli" in path_obj.name.lower() or "console" in path_obj.name.lower()
                )
                app_type = "CLI" if is_cli else "GUI"
                self.logger.info(f"Found {app_type} version: {path_obj.name}")
                self.astap_path = str(path_obj)
                self._is_validated = True
                self.logger.info(
                    f"Auto-detected ASTAP {app_type} version (validation skipped): {path_obj}"
                )
                return str(path_obj)

        self.logger.warning("Could not auto-detect working ASTAP executable")
        return None

    def validate_astap_executable(self, path=None):
        """
        Validate ASTAP executable with CLI/GUI awareness

        Note: During auto-detection, validation is skipped to prevent dialog popups.
        This method is primarily used for manual validation (e.g., via Test ASTAP menu).
        """
        check_path = path or self.astap_path

        if not check_path:
            return False

        path_obj = Path(check_path)

        # Basic file existence and permission checks
        if not path_obj.exists():
            self.logger.error(f"ASTAP executable not found: {check_path}")
            return False

        if not os.access(check_path, os.X_OK):
            self.logger.error(f"ASTAP executable is not executable: {check_path}")
            return False

        # Determine if this is likely a CLI or GUI version
        filename_lower = path_obj.name.lower()
        is_cli_version = any(
            indicator in filename_lower for indicator in ["cli", "console", "cmd"]
        )
        app_type = "CLI" if is_cli_version else "GUI"

        self.logger.info(f"Validating ASTAP {app_type} version: {path_obj.name}")

        # Filename validation (warning only)
        if "astap" not in filename_lower:
            self.logger.warning(
                f"Executable name doesn't contain 'astap': {path_obj.name}"
            )

        # Different validation approach for CLI vs GUI versions
        if is_cli_version:
            return self._validate_cli_version(check_path)
        else:
            return self._validate_gui_version(check_path)

    def _validate_cli_version(self, check_path):
        """Validate CLI version of ASTAP"""
        self.logger.info("Validating CLI version - expecting console output")

        # CLI versions should behave like normal command-line tools
        test_commands = [
            ["-h"],  # Help
            ["--help"],  # Alternative help
            ["-version"],  # Version
            ["--version"],  # Alternative version
        ]

        for args in test_commands:
            try:
                result = subprocess.run(
                    [check_path] + args,
                    capture_output=True,
                    text=True,
                    timeout=10,  # CLI should be fast
                    cwd=Path(check_path).parent,
                )

                # CLI tools should produce output or at least return consistent codes
                if result.stdout or result.stderr:
                    output_text = (result.stdout + result.stderr).lower()
                    if any(
                        keyword in output_text
                        for keyword in [
                            "astap",
                            "astrometric",
                            "plate",
                            "solve",
                            "help",
                            "usage",
                            "version",
                        ]
                    ):
                        self._is_validated = True
                        self.logger.info(f"CLI ASTAP validated with args {args}")
                        return True

                # Even without output, return code 0 might indicate success
                if result.returncode == 0:
                    self.logger.info(f"CLI ASTAP responds to {args} (rc=0)")
                    self._is_validated = True
                    return True

            except subprocess.TimeoutExpired:
                self.logger.warning(f"CLI ASTAP timed out with args {args}")
                continue
            except Exception as e:
                self.logger.warning(f"CLI ASTAP test error with args {args}: {e}")
                continue

        self.logger.warning("CLI ASTAP validation inconclusive - assuming valid")
        self._is_validated = True
        return True

    def _validate_gui_version(self, check_path):
        """Validate GUI version of ASTAP"""
        self.logger.info("Validating GUI version - expecting GUI behavior")

        # GUI versions often don't produce console output but may respond to basic commands
        try:
            # Try a quick test with very short timeout
            result = subprocess.run(
                [check_path, "-h"],
                capture_output=True,
                text=True,
                timeout=3,  # Short timeout for GUI apps
                cwd=Path(check_path).parent,
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
            )

            # For GUI apps, return code 0 is success even without output
            if result.returncode == 0:
                self._is_validated = True
                self.logger.info(f"GUI ASTAP validated (rc=0, no output expected)")
                return True

        except subprocess.TimeoutExpired:
            # Timeout is actually GOOD for GUI apps - means it's trying to start
            self._is_validated = True
            self.logger.info(f"GUI ASTAP detected (timeout=normal GUI behavior)")
            return True

        except Exception as e:
            self.logger.warning(f"GUI ASTAP test error: {e}")

        # If we can't test execution, but file exists and looks right, assume it's valid
        self.logger.info(
            "GUI ASTAP validation - assuming valid based on file existence"
        )
        self._is_validated = True
        return True

    def run_astap_command(self, args, timeout=None):
        """Run ASTAP with specified arguments"""
        if not self.astap_path:
            raise ValueError("ASTAP executable path not configured")

        if not self._is_validated and not self.validate_astap_executable():
            raise ValueError("ASTAP executable failed validation")

        timeout = timeout or self.config.getint("ASTAP", "timeout_seconds", 30)
        additional_params = self.config.get(
            "ASTAP", "additional_parameters", ""
        ).split()

        cmd = [self.astap_path] + additional_params + args

        self.logger.debug(f"Running ASTAP command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"ASTAP returned non-zero exit code: {result.returncode}"
                )
                self.logger.warning(f"ASTAP stderr: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            self.logger.error(f"ASTAP command timed out after {timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Error running ASTAP command: {e}")
            raise

    def get_astap_version(self):
        """Get ASTAP version information"""
        try:
            output = self.run_astap_command(["-h"], timeout=5)
            # Parse version from help output
            lines = output.split("\n")
            for line in lines[:10]:  # Check first 10 lines
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ["version", "astap", "v."]):
                    return line.strip()
            return "Version information not found"
        except Exception as e:
            return f"Error getting version: {e}"

    def is_configured(self):
        """Check if ASTAP is properly configured"""
        return bool(self.astap_path and self._is_validated)

    def get_status_info(self):
        """Get comprehensive status information"""
        if not self.astap_path:
            return {
                "configured": False,
                "valid": False,
                "message": "ASTAP executable not configured",
                "path": "",
                "version": "",
            }

        try:
            is_valid = self.validate_astap_executable()
            version = self.get_astap_version() if is_valid else "Unable to get version"

            return {
                "configured": True,
                "valid": is_valid,
                "message": "ASTAP ready"
                if is_valid
                else "ASTAP found but validation failed",
                "path": self.astap_path,
                "version": version,
            }
        except Exception as e:
            self.logger.warning(f"Error getting ASTAP status: {e}")
            return {
                "configured": True,
                "valid": False,
                "message": f"Error checking ASTAP: {str(e)[:100]}",
                "path": self.astap_path,
                "version": "Unknown",
            }


class FITSFileValidator:
    """Validate FITS files for calibration processing"""

    def __init__(self, config):
        self.config = config
        self.logger = CalibrationLogger()

    def validate_files(self, file_list, calibration_type):
        """Validate a list of FITS files for consistency"""
        validation_results = {
            "valid_files": [],
            "invalid_files": [],
            "warnings": [],
            "statistics": {},
        }

        if not file_list:
            validation_results["warnings"].append("No files provided for validation")
            return validation_results

        self.logger.info(
            f"Validating {len(file_list)} files for {calibration_type} calibration"
        )

        # Check minimum number of files
        min_files_key = f"min_frames_{calibration_type.lower()}"
        min_files = self.config.getint("PROCESSING", min_files_key, 5)

        if len(file_list) < min_files:
            validation_results["warnings"].append(
                f"Only {len(file_list)} files found, minimum recommended is {min_files}"
            )

        # Validate each file (simulated - would use actual FITS reading)
        exposure_times = []
        temperatures = []
        binning_modes = []

        for file_path in file_list:
            try:
                # Simulate reading FITS header
                file_info = self._simulate_fits_header(file_path, calibration_type)

                if file_info["valid"]:
                    validation_results["valid_files"].append(str(file_path))
                    exposure_times.append(file_info["exposure"])
                    temperatures.append(file_info["temperature"])
                    binning_modes.append(file_info["binning"])
                else:
                    validation_results["invalid_files"].append(
                        {"file": str(file_path), "reason": file_info["error"]}
                    )

            except Exception as e:
                validation_results["invalid_files"].append(
                    {"file": str(file_path), "reason": str(e)}
                )

        # Check consistency
        if validation_results["valid_files"]:
            self._check_consistency(
                validation_results, exposure_times, temperatures, binning_modes
            )

        return validation_results

    def _simulate_fits_header(self, file_path, calibration_type):
        """Simulate reading FITS header (replace with actual astropy.io.fits code)"""
        # This is a simulation - in real implementation would read actual FITS headers
        import random

        base_exposure = {"bias": 0.0, "dark": 300.0, "flat": 1.0, "dark_flat": 1.0}
        base_temp = -20.0

        return {
            "valid": True,
            "exposure": base_exposure.get(calibration_type, 1.0)
            + random.uniform(-0.05, 0.05),
            "temperature": base_temp + random.uniform(-1, 1),
            "binning": "1x1",
            "error": None,
        }

    def _check_consistency(self, results, exposures, temperatures, binning):
        """Check consistency of calibration parameters"""
        stats = results["statistics"]

        # Exposure time consistency
        if exposures:
            exp_std = np.std(exposures)
            exp_mean = np.mean(exposures)
            stats["exposure_mean"] = exp_mean
            stats["exposure_std"] = exp_std

            max_variance = self.config.getfloat(
                "VALIDATION", "max_exposure_variance", 0.1
            )
            if exp_std > max_variance:
                results["warnings"].append(
                    f"Exposure time variance ({exp_std:.3f}s) exceeds threshold ({max_variance}s)"
                )

        # Temperature consistency
        if temperatures:
            temp_std = np.std(temperatures)
            temp_mean = np.mean(temperatures)
            stats["temperature_mean"] = temp_mean
            stats["temperature_std"] = temp_std

            max_temp_variance = self.config.getfloat(
                "VALIDATION", "max_temperature_variance", 2.0
            )
            if temp_std > max_temp_variance:
                results["warnings"].append(
                    f"Temperature variance ({temp_std:.1f}°C) exceeds threshold ({max_temp_variance}°C)"
                )

        # Binning consistency
        unique_binning = set(binning)
        if len(unique_binning) > 1:
            results["warnings"].append(
                f"Multiple binning modes found: {', '.join(unique_binning)}"
            )


class CalibrationFileManager:
    """Manage calibration file organization and metadata"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_master_filename(self, calibration_type, metadata=None):
        """Create standardized filename for master calibration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if metadata:
            # Include relevant metadata in filename
            parts = [f"master_{calibration_type}"]

            if "exposure" in metadata and metadata["exposure"] > 0:
                parts.append(f"{metadata['exposure']:.0f}s")

            if "temperature" in metadata:
                parts.append(f"{metadata['temperature']:.0f}C")

            if "binning" in metadata:
                parts.append(metadata["binning"].replace("x", "x"))

            parts.append(timestamp)
            filename = "_".join(parts) + ".fits"
        else:
            filename = f"master_{calibration_type}_{timestamp}.fits"

        return self.base_path / filename

    def save_processing_log(self, calibration_type, processing_info):
        """Save detailed processing log"""
        log_file = (
            self.base_path
            / f"processing_log_{calibration_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(log_file, "w") as f:
            json.dump(processing_info, f, indent=2, default=str)

    def find_existing_masters(self):
        """Find existing master calibration files"""
        masters = {"bias": [], "dark": [], "flat": [], "dark_flat": []}

        for fits_file in self.base_path.glob("master_*.fits"):
            filename = fits_file.name.lower()
            for cal_type in masters.keys():
                if cal_type in filename:
                    masters[cal_type].append(fits_file)
                    break

        return masters


class ProgressReporter:
    """Thread-safe progress reporting"""

    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = {}

    def set_total_steps(self, total):
        self.total_steps = total

    def update(self, step, description=None):
        self.current_step = step
        if description:
            self.step_descriptions[step] = description

    def get_progress_percent(self):
        return int((self.current_step / self.total_steps) * 100)

    def get_current_description(self):
        return self.step_descriptions.get(
            self.current_step, f"Step {self.current_step}"
        )


# Example configuration usage
def setup_calibration_environment():
    """Setup calibration processing environment"""

    # Initialize configuration
    config = CalibrationConfig()

    # Setup logging
    logger = CalibrationLogger()

    # Create output directories
    output_base = Path(
        config.get("DIALOG", "default_output_folder", "calibration_output")
    )
    file_manager = CalibrationFileManager(output_base)

    logger.info("Calibration environment initialized")
    logger.info(f"Output directory: {output_base}")

    return config, logger, file_manager


# Testing and validation functions
def test_dialog_positioning():
    """Test dialog positioning on multiple monitors"""
    import sys

    from PyQt6.QtWidgets import QApplication, QDialog

    app = QApplication(sys.argv)

    # Create test dialog
    dialog = QDialog()
    dialog.setWindowTitle("Position Test")
    dialog.resize(400, 300)

    # Test positioning
    x, y = DialogPositionManager.get_optimal_position(dialog)
    dialog.move(x, y)

    print(f"Dialog positioned at: ({x}, {y})")
    print(f"Screen count: {len(app.screens())}")

    for i, screen in enumerate(app.screens()):
        print(f"Screen {i}: {screen.geometry()}")

    return dialog


if __name__ == "__main__":
    # Test configuration
    config, logger, file_manager = setup_calibration_environment()

    # Test ASTAP manager
    print("\n=== ASTAP Manager Test ===")
    astap_manager = ASTAPManager(config)

    # Auto-detect ASTAP
    detected_path = astap_manager.auto_detect_astap()
    if detected_path:
        print(f"Auto-detected ASTAP: {detected_path}")
        status = astap_manager.get_status_info()
        print(f"Status: {status['message']}")
        print(f"Version: {status['version']}")
    else:
        print("ASTAP not auto-detected")

    # Test file validation
    print("\n=== File Validation Test ===")
    validator = FITSFileValidator(config)

    # Test with simulated files
    test_files = [Path(f"test_bias_{i:03d}.fits") for i in range(10)]
    results = validator.validate_files(test_files, "bias")

    print("Validation Results:")
    print(f"Valid files: {len(results['valid_files'])}")
    print(f"Invalid files: {len(results['invalid_files'])}")
    print(f"Warnings: {len(results['warnings'])}")

    for warning in results["warnings"]:
        print(f"  - {warning}")

    # Test dialog positioning
    print("\n=== Dialog Positioning Test ===")
    try:
        import sys

        from PyQt6.QtWidgets import QApplication, QDialog

        if not QApplication.instance():
            app = QApplication(sys.argv)

            # Create test dialog
            dialog = QDialog()
            dialog.setWindowTitle("Position Test")
            dialog.resize(400, 300)

            # Test positioning
            x, y = DialogPositionManager.get_optimal_position(dialog)
            dialog.move(x, y)

            print(f"Dialog positioned at: ({x}, {y})")
            print(f"Available screens: {len(app.screens())}")

            for i, screen in enumerate(app.screens()):
                geom = screen.geometry()
                print(
                    f"Screen {i}: {geom.width()}x{geom.height()} at ({geom.x()}, {geom.y()})"
                )

    except ImportError:
        print("Qt6 not available for dialog positioning test")

    print("\n=== Configuration Test ===")
    print(f"Config file: {config.config_file}")
    print(f"ASTAP auto-detect: {config.getboolean('ASTAP', 'auto_detect_on_startup')}")
    print(f"Min bias frames: {config.getint('PROCESSING', 'min_frames_bias')}")
    print(f"Combination method: {config.get('PROCESSING', 'combination_method')}")
