#!/usr/bin/env python3
"""
PulseHunter Launcher Script
Simple script to launch the application with proper error handling
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    required = {
        "numpy": "numpy",
        "astropy": "astropy",
        "matplotlib": "matplotlib",
        "PyQt6": "PyQt6",
        "scipy": "scipy",
    }

    missing = []
    for name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(name)

    if missing:
        print("ERROR: Missing required packages!")
        print(f"Please install: {', '.join(missing)}")
        print("\nRun this command:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main launcher function"""
    print("=" * 60)
    print("PulseHunter - Optical SETI & Exoplanet Detection")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    try:
        # Import and run the main GUI
        from pulse_gui import main as run_gui

        print("Starting PulseHunter GUI...")
        run_gui()

    except Exception as e:
        print(f"\nERROR: Failed to start PulseHunter!")
        print(f"Error details: {e}")

        # Try to give helpful error messages
        if "pulse_gui" in str(e):
            print("\nMake sure pulse_gui.py is in the same directory as this script.")
        elif "pulsehunter_core" in str(e):
            print("\nMake sure pulsehunter_core.py is in the same directory.")
        elif "calibration_manager" in str(e):
            print("\nMake sure calibration_manager.py is in the same directory.")

        import traceback

        print("\nFull error trace:")
        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
