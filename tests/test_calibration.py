"""
Test suite for PulseHunter Enhanced Calibration System
Run this to verify your installation is working correctly
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        from calibration_dialog import CalibrationSetupDialog

        print("✓ calibration_dialog imported successfully")
    except ImportError as e:
        print(f"✗ Error importing calibration_dialog: {e}")
        return False

    try:
        from calibration_utilities import (
            ASTAPManager,
            CalibrationConfig,
            CalibrationLogger,
            DialogPositionManager,
        )

        print("✓ calibration_utilities imported successfully")
    except ImportError as e:
        print(f"✗ Error importing calibration_utilities: {e}")
        return False

    try:
        import numpy

        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Error importing numpy: {e}")
        return False

    try:
        from PyQt6.QtWidgets import QApplication

        print("✓ PyQt6 imported successfully")
    except ImportError as e:
        print(f"✗ Error importing PyQt6: {e}")
        return False

    return True


def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")

    try:
        from calibration_utilities import CalibrationConfig

        # Create temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            temp_config_file = f.name

        config = CalibrationConfig(temp_config_file)

        # Test basic operations
        assert config.get("PROCESSING", "combination_method") == "median"
        assert config.getint("PROCESSING", "min_frames_bias") == 10
        assert config.getboolean("ASTAP", "auto_detect_on_startup") == True

        # Test saving
        config.config.set("ASTAP", "executable_path", "/test/path/astap.exe")
        config.save_config()

        # Test loading
        config2 = CalibrationConfig(temp_config_file)
        assert config2.get("ASTAP", "executable_path") == "/test/path/astap.exe"

        # Cleanup
        os.unlink(temp_config_file)

        print("✓ Configuration system working correctly")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_astap_manager():
    """Test ASTAP manager functionality"""
    print("\nTesting ASTAP manager...")

    try:
        from calibration_utilities import ASTAPManager, CalibrationConfig

        config = CalibrationConfig()
        astap_manager = ASTAPManager(config)

        # Test basic properties
        initial_path = astap_manager.astap_path
        print(f"  Initial ASTAP path: {initial_path or 'Not set'}")

        # Test auto-detection (won't find anything in test environment)
        detected = astap_manager.auto_detect_astap()
        if detected:
            print(f"✓ ASTAP auto-detected at: {detected}")
        else:
            print("  No ASTAP installation auto-detected (this is normal for testing)")

        # Test validation with invalid path
        is_valid = astap_manager.validate_astap_executable("/invalid/path/astap.exe")
        assert not is_valid, "Invalid path should not validate"

        # Test status info
        status = astap_manager.get_status_info()
        assert isinstance(status, dict)
        assert "configured" in status
        assert "valid" in status

        print("✓ ASTAP manager working correctly")
        return True

    except Exception as e:
        print(f"✗ ASTAP manager test failed: {e}")
        return False


def test_file_validator():
    """Test FITS file validator"""
    print("\nTesting file validator...")

    try:
        from calibration_utilities import CalibrationConfig, FITSFileValidator

        config = CalibrationConfig()
        validator = FITSFileValidator(config)

        # Create some dummy file paths
        test_files = [Path(f"test_bias_{i:03d}.fits") for i in range(5)]

        # Test validation
        results = validator.validate_files(test_files, "bias")

        assert isinstance(results, dict)
        assert "valid_files" in results
        assert "invalid_files" in results
        assert "warnings" in results
        assert "statistics" in results

        print("✓ File validator working correctly")
        return True

    except Exception as e:
        print(f"✗ File validator test failed: {e}")
        return False


def test_logger():
    """Test logging system"""
    print("\nTesting logging system...")

    try:
        from calibration_utilities import CalibrationLogger

        logger = CalibrationLogger()

        # Test logging methods
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.debug("Test debug message")

        # Check if log directory was created
        log_dir = Path("logs")
        if log_dir.exists():
            print("✓ Log directory created")
            # Find log files
            log_files = list(log_dir.glob("calibration_*.log"))
            if log_files:
                print(f"✓ Log file created: {log_files[0].name}")
            else:
                print("  No log files found (may be normal)")
        else:
            print("  Log directory not created (may be normal)")

        print("✓ Logging system working correctly")
        return True

    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def test_dialog_creation():
    """Test dialog creation without showing"""
    print("\nTesting dialog creation...")

    try:
        # Create QApplication if it doesn't exist
        app = QApplication.instance() or QApplication(sys.argv)

        from calibration_dialog import CalibrationSetupDialog

        # Create dialog (but don't show it)
        dialog = CalibrationSetupDialog()

        # Test basic properties
        assert dialog.windowTitle() == "PulseHunter - Calibration Setup"
        assert dialog.minimumSize().width() >= 800
        assert dialog.minimumSize().height() >= 700

        # Test that key components exist
        assert hasattr(dialog, "astap_path_edit")
        assert hasattr(dialog, "tab_widget")
        assert hasattr(dialog, "progress_bar")
        assert hasattr(dialog, "log_display")

        print("✓ Dialog creation working correctly")
        return True

    except Exception as e:
        print(f"✗ Dialog creation test failed: {e}")
        return False


def test_settings_persistence():
    """Test settings persistence"""
    print("\nTesting settings persistence...")

    try:
        # Use temporary settings
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)

        # Test settings save/load
        settings = QSettings("PulseHunterTest", "CalibrationTest")
        settings.setValue("test_key", "test_value")
        settings.setValue("test_int", 42)
        settings.setValue("test_bool", True)

        # Create new settings instance
        settings2 = QSettings("PulseHunterTest", "CalibrationTest")

        assert settings2.value("test_key") == "test_value"
        assert settings2.value("test_int", type=int) == 42
        assert settings2.value("test_bool", type=bool) == True

        # Cleanup
        settings.clear()

        print("✓ Settings persistence working correctly")
        return True

    except Exception as e:
        print(f"✗ Settings persistence test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")

    required_files = [
        "calibration_dialog.py",
        "calibration_utilities.py",
        "pulse_gui.py",
    ]

    optional_files = ["calibration_config.ini", "requirements.txt"]

    all_good = True

    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} missing (REQUIRED)")
            all_good = False

    for file in optional_files:
        if Path(file).exists():
            print(f"✓ {file} found")
        else:
            print(f"  {file} missing (optional - will be created)")

    return all_good


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("PulseHunter Enhanced Calibration System Test Suite")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Configuration System", test_configuration),
        ("ASTAP Manager", test_astap_manager),
        ("File Validator", test_file_validator),
        ("Logging System", test_logger),
        ("Dialog Creation", test_dialog_creation),
        ("Settings Persistence", test_settings_persistence),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * (len(test_name) + 1))

        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")

    if failed == 0:
        print("\n🎉 All tests passed! Your PulseHunter installation is ready.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False


def main():
    """Main test runner"""
    success = run_all_tests()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    if success:
        print("1. Run 'python pulse_gui.py' to start PulseHunter")
        print("2. Configure ASTAP using the Calibration menu")
        print("3. Set up your calibration frames")
        print("4. Start processing your astronomical images!")
    else:
        print("1. Fix any import errors by installing missing dependencies")
        print("2. Ensure all required files are in the correct location")
        print("3. Re-run this test script to verify fixes")
        print("4. Check the integration guide for troubleshooting tips")

    print("\nFor help, see the integration guide or check the logs directory.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
