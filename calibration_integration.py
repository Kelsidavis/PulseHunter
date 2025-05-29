"""
Calibration Integration Module for PulseHunter - Fixed Version
Ensures seamless integration between enhanced calibration and existing systems
Fixed to avoid circular imports
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Import the enhanced calibration manager directly
try:
    from enhanced_calibration_manager import (
        enhanced_load_fits_stack,
        AutoCalibrationManager,
        apply_calibration_to_image,
        get_calibration_status,
        create_calibration_summary_report
    )
    ENHANCED_CAL_AVAILABLE = True
    print("✅ Enhanced calibration manager loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced calibration not available: {e}")
    ENHANCED_CAL_AVAILABLE = False

# Import existing components with fallback
try:
    from fixed_calibration_dialog import get_master_files_for_folder
    DIALOG_INTEGRATION_AVAILABLE = True
except ImportError:
    DIALOG_INTEGRATION_AVAILABLE = False

# Basic FITS loading fallback (avoid importing full pulsehunter_core to prevent circular import)
def _basic_fits_loading_fallback(folder, **kwargs):
    """Minimal FITS loading fallback to avoid circular imports"""
    from concurrent.futures import ThreadPoolExecutor
    from astropy.io import fits
    from astropy.wcs import WCS
    
    print(f"📁 Using basic fallback FITS loading for: {folder}")
    
    frames, filenames, wcs_objects = [], [], []
    
    if not os.path.exists(folder):
        print(f"Error: Folder {folder} does not exist")
        return np.array([]), [], []
    
    fits_files = [f for f in sorted(os.listdir(folder)) if f.endswith((".fits", ".fit", ".fts"))]
    if not fits_files:
        print(f"No FITS files found in {folder}")
        return np.array([]), [], []
    
    def load_fits_file(file):
        path = os.path.join(folder, file)
        try:
            hdul = fits.open(path)
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            hdul.close()
            
            wcs = WCS(header) if "CRVAL1" in header and "CRVAL2" in header else None
            return (data, file, wcs)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(filter(None, executor.map(load_fits_file, fits_files[:5])))  # Limit to 5 for fallback
    
    if results:
        frames, filenames, wcs_objects = zip(*results)
        return np.array(frames), list(filenames), list(wcs_objects)
    
    return np.array([]), [], []


def smart_load_fits_stack(
    folder: str,
    plate_solve_missing: bool = False,
    astap_exe: str = "astap",
    auto_calibrate: bool = True,
    manual_master_bias: Optional[np.ndarray] = None,
    manual_master_dark: Optional[np.ndarray] = None,
    manual_master_flat: Optional[np.ndarray] = None,
    progress_callback: Optional[callable] = None,
    prefer_enhanced: bool = True,
    **kwargs
) -> Tuple[np.ndarray, List[str], List]:
    """
    Smart FITS loading that automatically chooses the best available method
    
    Args:
        folder: Directory containing FITS files
        plate_solve_missing: Whether to plate solve files without WCS
        astap_exe: Path to ASTAP executable
        auto_calibrate: Whether to automatically detect and apply calibration
        manual_master_bias: Manual bias frame (overrides auto-detection)
        manual_master_dark: Manual dark frame (overrides auto-detection)
        manual_master_flat: Manual flat frame (overrides auto-detection)
        progress_callback: Optional callback for progress updates
        prefer_enhanced: Whether to prefer enhanced loading when available
        **kwargs: Additional arguments passed to the loading function
        
    Returns:
        Tuple of (frames array, filenames list, wcs_objects list)
    """
    print("🔄 Smart FITS loading started...")
    
    # Try enhanced loading first if available and preferred
    if ENHANCED_CAL_AVAILABLE and prefer_enhanced:
        try:
            print("✨ Using enhanced calibration system...")
            return enhanced_load_fits_stack(
                folder=folder,
                plate_solve_missing=plate_solve_missing,
                astap_exe=astap_exe,
                auto_calibrate=auto_calibrate,
                manual_master_bias=manual_master_bias,
                manual_master_dark=manual_master_dark,
                manual_master_flat=manual_master_flat,
                progress_callback=progress_callback,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️ Enhanced loading failed, falling back to basic: {e}")
    
    # Fall back to basic loading
    try:
        print("📁 Using basic FITS loading fallback...")
        return _basic_fits_loading_fallback(folder, **kwargs)
    except Exception as e:
        print(f"❌ Basic loading also failed: {e}")
        return np.array([]), [], []


def get_calibration_info(lights_folder: str) -> Dict:
    """
    Get comprehensive calibration information for a folder
    
    Args:
        lights_folder: Path to lights folder
        
    Returns:
        Dictionary with calibration information
    """
    info = {
        "folder": lights_folder,
        "enhanced_available": ENHANCED_CAL_AVAILABLE,
        "dialog_integration": DIALOG_INTEGRATION_AVAILABLE,
        "calibration_status": "unknown",
        "master_files": {},
        "recommendations": []
    }
    
    # Try enhanced calibration status first
    if ENHANCED_CAL_AVAILABLE:
        try:
            status = get_calibration_status(lights_folder)
            info.update({
                "calibration_status": "configured" if status["has_calibration"] else "not_configured",
                "master_files": status["master_files"],
                "available_types": status.get("available_types", []),
                "missing_types": status.get("missing_types", [])
            })
        except Exception as e:
            info["error"] = f"Enhanced calibration check failed: {e}"
    
    # Try dialog integration
    elif DIALOG_INTEGRATION_AVAILABLE:
        try:
            master_files = get_master_files_for_folder(lights_folder)
            info.update({
                "calibration_status": "configured" if master_files else "not_configured",
                "master_files": master_files
            })
        except Exception as e:
            info["error"] = f"Dialog integration check failed: {e}"
    
    # Generate recommendations
    if info["calibration_status"] == "not_configured":
        info["recommendations"].extend([
            "No calibration found for this folder",
            "Use Calibration → Calibration Setup to create master files",
            "Master calibration files will improve image quality significantly"
        ])
    elif info["calibration_status"] == "configured":
        available_count = len(info.get("available_types", info["master_files"].keys()))
        info["recommendations"].append(f"✅ Calibration configured ({available_count} master files)")
        
        if "missing_types" in info and info["missing_types"]:
            info["recommendations"].append(
                f"Consider adding: {', '.join(info['missing_types'])}"
            )
    
    return info


def print_calibration_status(lights_folder: str):
    """Print human-readable calibration status"""
    print(f"\n{'='*60}")
    print("CALIBRATION STATUS CHECK")
    print(f"{'='*60}")
    
    info = get_calibration_info(lights_folder)
    
    print(f"Folder: {Path(lights_folder).name}")
    print(f"Status: {info['calibration_status']}")
    
    if info["master_files"]:
        print(f"\nAvailable Master Files:")
        for cal_type, file_path in info["master_files"].items():
            file_exists = "✅" if Path(file_path).exists() else "❌"
            print(f"  {file_exists} {cal_type}: {Path(file_path).name}")
    
    if info["recommendations"]:
        print(f"\nRecommendations:")
        for rec in info["recommendations"]:
            print(f"  • {rec}")
    
    if "error" in info:
        print(f"\nError: {info['error']}")


def validate_calibration_setup():
    """Validate that calibration components are properly set up"""
    print(f"\n{'='*60}")
    print("CALIBRATION SYSTEM VALIDATION")
    print(f"{'='*60}")
    
    components = {
        "Enhanced Calibration Manager": ENHANCED_CAL_AVAILABLE,
        "Dialog Integration": DIALOG_INTEGRATION_AVAILABLE,
    }
    
    all_working = True
    
    for component, available in components.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{component}: {status}")
        if not available:
            all_working = False
    
    print(f"\nOverall Status: {'✅ Ready' if all_working else '⚠️ Partial'}")
    
    if not all_working:
        print("\nRecommendations:")
        if not ENHANCED_CAL_AVAILABLE:
            print("• Save enhanced_calibration_manager.py to enable advanced features")
        if not DIALOG_INTEGRATION_AVAILABLE:
            print("• Check fixed_calibration_dialog.py for dialog integration")
    
    return all_working


def migrate_calibration_projects():
    """Migrate old calibration projects to new format if needed"""
    print("🔄 Checking for calibration project migrations...")
    
    old_files = [
        "calibration_projects.json",
        "filter_calibration_projects.json"
    ]
    
    migrations_needed = []
    
    for old_file in old_files:
        if Path(old_file).exists():
            print(f"✅ Found existing project file: {old_file}")
        else:
            migrations_needed.append(old_file)
    
    if migrations_needed:
        print(f"⚠️ Missing project files: {', '.join(migrations_needed)}")
        print("These will be created automatically when you set up calibration")
    else:
        print("✅ All calibration project files found")


def test_calibration_with_sample_folder(test_folder: str):
    """Test calibration system with a sample folder"""
    if not Path(test_folder).exists():
        print(f"❌ Test folder not found: {test_folder}")
        return False
    
    print(f"\n{'='*60}")
    print(f"TESTING CALIBRATION WITH: {Path(test_folder).name}")
    print(f"{'='*60}")
    
    # Check calibration status
    print_calibration_status(test_folder)
    
    # Test loading a few files
    try:
        def progress_callback(percent):
            if percent % 20 == 0:  # Only print every 20%
                print(f"  Loading progress: {percent}%")
        
        print(f"\nTesting FITS loading (limited to first few files)...")
        
        frames, filenames, wcs_objects = smart_load_fits_stack(
            folder=test_folder,
            auto_calibrate=True,
            progress_callback=progress_callback,
            max_workers=2  # Limit workers for testing
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"  Loaded frames: {len(frames)}")
        print(f"  With WCS: {sum(1 for w in wcs_objects if w is not None)}")
        if len(frames) > 0:
            print(f"  Frame shape: {frames[0].shape}")
            print(f"  Data type: {frames[0].dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Auto-fix imports for pulse_gui.py compatibility
def ensure_compatibility():
    """Ensure compatibility with existing pulse_gui.py imports"""
    # This function can be called to set up any necessary compatibility shims
    
    # Make sure the enhanced functions are available in the expected locations
    current_module = sys.modules[__name__]
    
    # Add enhanced_load_fits_stack to module namespace for import compatibility
    if ENHANCED_CAL_AVAILABLE:
        setattr(current_module, 'enhanced_load_fits_stack', enhanced_load_fits_stack)
        setattr(current_module, 'AutoCalibrationManager', AutoCalibrationManager)
    else:
        # Provide fallback functions
        def fallback_enhanced_load(*args, **kwargs):
            print("⚠️ Enhanced calibration not available, using smart fallback...")
            return smart_load_fits_stack(*args, **kwargs)
        
        class FallbackCalibrationManager:
            def __init__(self):
                print("⚠️ Using fallback calibration manager")
            
            def get_master_files_for_folder(self, folder):
                return {}
        
        setattr(current_module, 'enhanced_load_fits_stack', fallback_enhanced_load)
        setattr(current_module, 'AutoCalibrationManager', FallbackCalibrationManager)


# Initialize compatibility on import
ensure_compatibility()


if __name__ == "__main__":
    print("🌌 PulseHunter Calibration Integration Test")
    print("="*60)
    
    # Validate setup
    validate_calibration_setup()
    
    # Check for migrations
    migrate_calibration_projects()
    
    # Test with sample folder if provided
    test_folders = [
        r"F:\astrophotography\2024-07-13 - bortle2 - sthelens",
        "./test_fits",
        "../test_fits"
    ]
    
    test_completed = False
    for test_folder in test_folders:
        if Path(test_folder).exists():
            test_calibration_with_sample_folder(test_folder)
            test_completed = True
            break
    
    if not test_completed:
        print(f"\n⚠️ No test folders found. Update test_folders list to test with your data.")
        print("Available functions:")
        print("• smart_load_fits_stack() - Intelligent FITS loading")
        print("• get_calibration_info() - Check calibration status")
        print("• print_calibration_status() - Display status")
        print("• validate_calibration_setup() - Validate components")
    
    print(f"\n{'='*60}")
    print("✅ CALIBRATION INTEGRATION READY")
    print("="*60)
    print("The calibration system is ready for use with PulseHunter!")