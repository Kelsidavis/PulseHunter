#!/usr/bin/env python3
"""
PulseHunter Application Launcher
Handles environment setup, error checking, and graceful startup
"""

import importlib.util
import os
import sys
import traceback
from pathlib import Path


def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade your Python installation")
        return False

    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")

    required_modules = [
        ("PyQt6", "PyQt6.QtWidgets"),
        ("numpy", "numpy"),
        ("pathlib", "pathlib"),
        ("configparser", "configparser"),
    ]

    optional_modules = [
        ("astropy", "astropy"),
        ("scipy", "scipy"),
        ("PIL", "PIL"),
    ]

    missing_required = []
    missing_optional = []

    # Check required modules
    for name, import_name in required_modules:
        try:
            importlib.import_module(import_name)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} (REQUIRED)")
            missing_required.append(name)

    # Check optional modules
    for name, import_name in optional_modules:
        try:
            importlib.import_module(import_name)
            print(f"✓ {name}")
        except ImportError:
            print(f"⚠️  {name} (optional)")
            missing_optional.append(name)

    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        print("Install them with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        print(
            "For full functionality, install with: pip install "
            + " ".join(missing_optional)
        )

    return True


def check_files():
    """Check if all required PulseHunter files are present"""
    print("\nChecking PulseHunter files...")

    required_files = [
        "pulse_gui.py",
        "fixed_calibration_dialog.py",
        "calibration_utilities.py",
    ]

    missing_files = []

    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"❌ {file} (REQUIRED)")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all PulseHunter files are in the current directory")
        return False

    return True


def setup_environment():
    """Setup the runtime environment"""
    print("\nSetting up environment...")

    # Create directories if they don't exist
    directories = ["logs", "calibration_output", "temp"]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

    # Set environment variables
    os.environ["PULSEHUNTER_HOME"] = str(Path.cwd())

    return True


def show_system_info():
    """Display system information"""
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {Path.cwd()}")
    print(f"PulseHunter Home: {os.environ.get('PULSEHUNTER_HOME', 'Not set')}")

    # Try to get Qt version
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

        print(f"Qt Version: {QT_VERSION_STR}")
        print(f"PyQt6 Version: {PYQT_VERSION_STR}")
    except ImportError:
        print("Qt Version: Not available")


def launch_application():
    """Launch the main PulseHunter application"""
    print("\n" + "=" * 50)
    print("LAUNCHING PULSEHUNTER")
    print("=" * 50)

    try:
        # Import and run the main application
        from pulse_gui import main

        print("✓ Starting PulseHunter GUI...")
        main()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(
            "Please check that all PulseHunter files are present and dependencies are installed"
        )
        return False
    except Exception as e:
        print(f"❌ Application error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

    return True


def run_diagnostics():
    """Run diagnostic tests"""
    print("=" * 50)
    print("PULSEHUNTER DIAGNOSTIC CHECK")
    print("=" * 50)

    # Run test suite if available
    try:
        from test_calibration import run_all_tests

        print("Running diagnostic tests...\n")
        success = run_all_tests()

        if success:
            print("\n✓ All diagnostic tests passed")
            return True
        else:
            print("\n❌ Some diagnostic tests failed")
            return False

    except ImportError:
        print("⚠️  Diagnostic test suite not available")
        print("Basic checks will be performed instead...\n")

        # Run basic checks
        checks = [
            check_python_version,
            check_dependencies,
            check_files,
            setup_environment,
        ]

        for check in checks:
            if not check():
                return False

        return True


def show_help():
    """Show help information"""
    print(
        """
PulseHunter Application Launcher

Usage: python launch_pulsehunter.py [OPTIONS]

Options:
  --help, -h     Show this help message
  --test, -t     Run diagnostic tests only
  --info, -i     Show system information only
  --version, -v  Show version information
  --debug, -d    Enable debug mode

Examples:
  python launch_pulsehunter.py          # Normal startup
  python launch_pulsehunter.py --test   # Run tests only
  python launch_pulsehunter.py --info   # Show system info

For more information, visit: https://github.com/Kelsidavis/PulseHunter
    """
    )


def show_version():
    """Show version information"""
    print(
        """
PulseHunter Enhanced Calibration System
Version: Alpha (Enhanced)
Author: Kelsi Davis
Website: https://geekastro.dev
GitHub: https://github.com/Kelsidavis/PulseHunter

This version includes:
- Enhanced calibration dialog with progress tracking
- ASTAP integration with auto-detection
- Dark flat frame support
- Existing master file usage
- Consistent dialog positioning
- Comprehensive logging and error handling
    """
    )


def main():
    """Main launcher function"""
    # Parse command line arguments
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        show_help()
        return 0

    if "--version" in args or "-v" in args:
        show_version()
        return 0

    if "--info" in args or "-i" in args:
        show_system_info()
        return 0

    debug_mode = "--debug" in args or "-d" in args
    test_only = "--test" in args or "-t" in args

    if debug_mode:
        print("🐛 Debug mode enabled")
        os.environ["PULSEHUNTER_DEBUG"] = "1"

    print("🌌 PulseHunter Enhanced Calibration System")
    print("   Optical SETI & Exoplanet Detection Pipeline")
    print()

    # Run diagnostics
    if not run_diagnostics():
        print("\n❌ Pre-flight checks failed!")
        print("Please fix the errors above before launching PulseHunter")
        return 1

    if test_only:
        print("\n✓ Diagnostic tests completed successfully")
        return 0

    # Show system information
    if debug_mode:
        show_system_info()

    # Launch the application
    print("\n🚀 All checks passed! Launching PulseHunter...")

    try:
        if launch_application():
            print("\n✓ PulseHunter closed successfully")
            return 0
        else:
            print("\n❌ PulseHunter encountered an error")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️  Launch interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if debug_mode:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
