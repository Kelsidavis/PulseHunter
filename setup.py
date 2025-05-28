#!/usr/bin/env python3
"""
PulseHunter Enhanced Calibration System Setup Script
Handles installation, configuration, and initial setup
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Setup metadata
PULSEHUNTER_VERSION = "Alpha-Enhanced"
REQUIRED_PYTHON = (3, 8)
REQUIRED_MODULES = ["PyQt6>=6.4.0", "numpy>=1.21.0", "configparser"]
OPTIONAL_MODULES = ["astropy>=5.0.0", "scipy>=1.7.0", "Pillow>=8.0.0"]


def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")

    if sys.version_info < REQUIRED_PYTHON:
        print(
            f"❌ Error: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} or higher is required"
        )
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
        return False

    print(
        f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return True


def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling dependencies...")

    # Required packages
    print("Installing required packages...")
    for package in REQUIRED_MODULES:
        print(f"  Installing {package}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"  ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            print(f"     Output: {e.stdout}")
            print(f"     Error: {e.stderr}")
            return False

    # Optional packages
    print("\nInstalling optional packages...")
    for package in OPTIONAL_MODULES:
        print(f"  Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"  ✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} installation failed (optional)")

    return True


def create_directory_structure():
    """Create required directories"""
    print("\nCreating directory structure...")

    directories = [
        "logs",
        "calibration_output",
        "temp",
        "resources",
        "examples",
        "docs",
    ]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"  Directory already exists: {directory}")

    return True


def create_configuration_files():
    """Create initial configuration files"""
    print("\nCreating configuration files...")

    # Create calibration_config.ini if it doesn't exist
    config_file = Path("calibration_config.ini")
    if not config_file.exists():
        config_content = """[PROCESSING]
min_frames_bias = 10
min_frames_dark = 5
min_frames_flat = 5
min_frames_dark_flat = 5
sigma_clipping_threshold = 3.0
combination_method = median
output_format = fits
compress_output = True
save_statistics = True

[VALIDATION]
check_exposure_consistency = True
max_exposure_variance = 0.1
check_temperature_consistency = True
max_temperature_variance = 2.0
check_binning_consistency = True
required_keywords = EXPTIME,IMAGETYP,CCD-TEMP

[DIALOG]
remember_positions = True
auto_save_settings = True
show_advanced_options = False
default_output_folder = calibration_output
auto_find_calibration_folders = True

[ASTAP]
executable_path =
auto_detect_on_startup = True
timeout_seconds = 30
additional_parameters =
verify_on_load = True
"""
        with open(config_file, "w") as f:
            f.write(config_content)
        print("✓ Created calibration_config.ini")
    else:
        print("  calibration_config.ini already exists")

    # Create desktop shortcut template
    if sys.platform == "win32":
        create_windows_shortcut()
    elif sys.platform == "linux":
        create_linux_desktop_entry()

    return True


def create_windows_shortcut():
    """Create Windows desktop shortcut"""
    try:
        import winshell
        from win32com.client import Dispatch

        desktop = winshell.desktop()
        shortcut_path = Path(desktop) / "PulseHunter.lnk"

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = str(Path.cwd() / "launch_pulsehunter.py")
        shortcut.WorkingDirectory = str(Path.cwd())
        shortcut.IconLocation = str(Path.cwd() / "resources" / "icon.ico")
        shortcut.save()

        print("✓ Created desktop shortcut")
    except ImportError:
        print("  Desktop shortcut creation requires pywin32 (optional)")
    except Exception as e:
        print(f"  Could not create desktop shortcut: {e}")


def create_linux_desktop_entry():
    """Create Linux desktop entry"""
    try:
        desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=PulseHunter
Comment=Optical SETI & Exoplanet Detection Pipeline
Exec={sys.executable} {Path.cwd() / "launch_pulsehunter.py"}
Icon={Path.cwd() / "resources" / "icon.png"}
Terminal=false
Categories=Science;Astronomy;
"""

        # Try to create in user's applications directory
        apps_dir = Path.home() / ".local" / "share" / "applications"
        if apps_dir.exists():
            desktop_file = apps_dir / "pulsehunter.desktop"
            with open(desktop_file, "w") as f:
                f.write(desktop_entry)
            os.chmod(desktop_file, 0o755)
            print("✓ Created desktop entry")
        else:
            print("  Could not create desktop entry (applications directory not found)")
    except Exception as e:
        print(f"  Could not create desktop entry: {e}")


def download_astap_info():
    """Provide information about downloading ASTAP"""
    print("\n" + "=" * 60)
    print("ASTAP PLATE SOLVING SOFTWARE")
    print("=" * 60)
    print("PulseHunter requires ASTAP for plate solving and astrometric calibration.")
    print()
    print("ASTAP is not automatically installed and must be downloaded separately:")
    print("  📥 Download from: https://www.hnsky.org/astap.htm")
    print()
    print("🎯 RECOMMENDED INSTALLATION:")
    print("  Windows: Install to C:/Program Files/astap/ (default location)")
    print("  Linux:   Install to /usr/local/bin/ or use package manager")
    print("  macOS:   Use the macOS package or Homebrew")
    print()
    print("Installation instructions:")
    print("  Windows: Download and run the Windows installer")
    print("           → Accept default installation path for auto-detection")
    print("  Linux:   Download the Linux package or compile from source")
    print("  macOS:   Use the macOS package")
    print()
    print("After installing ASTAP:")
    print("  1. Launch PulseHunter")
    print("  2. ASTAP should be auto-detected at startup")
    print("  3. If not detected, use Calibration → Configure ASTAP")
    print("  4. Test the connection with Calibration → Test ASTAP")
    print()
    print(
        "💡 Pro tip: Install to C:/Program Files/astap/astap.exe for instant detection!"
    )

    return True


def create_example_files():
    """Create example files and documentation"""
    print("\nCreating example files...")

    examples_dir = Path("examples")

    # Create example configuration
    example_config = examples_dir / "example_config.ini"
    with open(example_config, "w") as f:
        f.write(
            """# Example PulseHunter Configuration
# Copy this to calibration_config.ini and modify as needed

[PROCESSING]
# Minimum number of frames required for each calibration type
min_frames_bias = 20      # More bias frames = better noise reduction
min_frames_dark = 10      # Dark frames for thermal noise
min_frames_flat = 10      # Flat frames for vignetting correction
min_frames_dark_flat = 5  # Dark flats for flat field correction

# Frame combination method
combination_method = median  # Options: median, mean, sigma_clipped_mean
sigma_clipping_threshold = 3.0

[ASTAP]
# ASTAP executable path (auto-detected if possible)
executable_path = C:/Program Files/astap/astap.exe
timeout_seconds = 60

[DIALOG]
# Default output folder for master calibration files
default_output_folder = ./masters
"""
        )

    # Create README file
    readme_file = Path("README_Enhanced.md")
    with open(readme_file, "w") as f:
        f.write(
            f"""# PulseHunter Enhanced Calibration System

Version: {PULSEHUNTER_VERSION}

## Overview

This enhanced version of PulseHunter includes:
- Improved calibration dialog with progress tracking
- ASTAP integration with auto-detection
- Dark flat frame support
- Existing master file usage
- Consistent dialog positioning
- Comprehensive logging

## Quick Start

1. **Install ASTAP**: Download from https://www.hnsky.org/astap.htm
2. **Launch PulseHunter**: Run `python launch_pulsehunter.py`
3. **Configure ASTAP**: Use Calibration → Configure ASTAP
4. **Setup Calibration**: Use Calibration → Calibration Setup
5. **Process Images**: Use Processing → Process Images

## File Structure

- `pulse_gui.py` - Main application
- `calibration_dialog.py` - Enhanced calibration dialog
- `calibration_utilities.py` - Supporting utilities
- `fits_processing.py` - FITS file processing
- `launch_pulsehunter.py` - Application launcher
- `test_calibration.py` - Test suite
- `calibration_config.ini` - Configuration file

## Configuration

Edit `calibration_config.ini` to customize:
- Processing parameters
- ASTAP settings
- Dialog preferences
- Validation thresholds

## Troubleshooting

- Run `python test_calibration.py` to check installation
- Check logs in the `logs/` directory
- Use `python launch_pulsehunter.py --debug` for verbose output

## Support

- GitHub: https://github.com/Kelsidavis/PulseHunter
- Website: https://geekastro.dev
- Email: pulsehunter@geekastro.dev
"""
        )

    print("✓ Created example files and documentation")
    return True


def verify_installation():
    """Verify that installation completed successfully"""
    print("\nVerifying installation...")

    required_files = [
        "pulse_gui.py",
        "calibration_dialog.py",
        "calibration_utilities.py",
        "launch_pulsehunter.py",
        "calibration_config.ini",
    ]

    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} (MISSING)")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ Installation incomplete. Missing files: {', '.join(missing_files)}")
        return False

    # Test import
    try:
        print("\nTesting module imports...")
        import calibration_dialog

        import calibration_utilities

        print("✓ All modules import successfully")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

    print("\n✓ Installation verification completed successfully!")
    return True


def run_setup():
    """Run the complete setup process"""
    print("=" * 60)
    print(f"PULSEHUNTER ENHANCED CALIBRATION SETUP v{PULSEHUNTER_VERSION}")
    print("=" * 60)
    print("Setting up PulseHunter Enhanced Calibration System...")
    print()

    setup_steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directory_structure),
        ("Creating configuration", create_configuration_files),
        ("Creating examples", create_example_files),
        ("Verifying installation", verify_installation),
    ]

    completed_steps = 0

    for step_name, step_func in setup_steps:
        print(f"\n{step_name}:")
        print("-" * (len(step_name) + 1))

        try:
            if step_func():
                completed_steps += 1
                print(f"✓ {step_name} completed")
            else:
                print(f"❌ {step_name} failed")
                break
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            break

    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"Completed steps: {completed_steps}/{len(setup_steps)}")

    if completed_steps == len(setup_steps):
        print("\n🎉 Setup completed successfully!")
        download_astap_info()
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Download and install ASTAP from https://www.hnsky.org/astap.htm")
        print("2. Run: python launch_pulsehunter.py")
        print("3. Configure ASTAP using the Calibration menu")
        print("4. Start processing your astronomical images!")
        print()
        print("For help and documentation, see README_Enhanced.md")
        return True
    else:
        print(
            f"\n⚠️  Setup incomplete ({completed_steps}/{len(setup_steps)} steps completed)"
        )
        print("Please fix the errors above and run setup.py again")
        return False


def show_help():
    """Show help information"""
    print(
        f"""
PulseHunter Enhanced Calibration Setup v{PULSEHUNTER_VERSION}

Usage: python setup.py [OPTION]

Options:
  --help, -h     Show this help
  --deps-only    Install dependencies only
  --verify       Verify existing installation
  --clean        Clean installation (remove generated files)

Examples:
  python setup.py              # Full setup
  python setup.py --deps-only  # Install dependencies only
  python setup.py --verify     # Check installation
"""
    )


def clean_installation():
    """Clean up installation files"""
    print("Cleaning installation...")

    files_to_remove = ["calibration_config.ini", "README_Enhanced.md"]

    dirs_to_remove = ["logs", "temp", "examples", "__pycache__"]

    for file in files_to_remove:
        file_path = Path(file)
        if file_path.exists():
            file_path.unlink()
            print(f"✓ Removed {file}")

    for directory in dirs_to_remove:
        dir_path = Path(directory)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✓ Removed directory {directory}")

    print("✓ Cleanup completed")


def main():
    """Main setup function"""
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        show_help()
        return 0

    if "--clean" in args:
        clean_installation()
        return 0

    if "--verify" in args:
        return 0 if verify_installation() else 1

    if "--deps-only" in args:
        if check_python_version() and install_dependencies():
            print("✓ Dependencies installed successfully")
            return 0
        else:
            print("❌ Dependency installation failed")
            return 1

    # Run full setup
    return 0 if run_setup() else 1


if __name__ == "__main__":
    sys.exit(main())
