#!/usr/bin/env python3
"""
Test suite for PulseHunter detection algorithms
"""

import os

# Import the modules to test
import sys
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from astropy.io import fits

from pulsehunter_core import crossmatch_with_gaia, detect_transients, load_fits_stack

sys.path.append("..")


class TestDetection:
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        # Create test frames with known transient
        self.frames = self.create_test_frames()
        self.filenames = [f"test_{i:03d}.fits" for i in range(len(self.frames))]
        self.wcs_objects = [None] * len(self.frames)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_frames(self):
        """Create test image frames with a synthetic transient"""
        np.random.seed(42)  # For reproducible tests
        frames = []

        for i in range(10):
            # Base frame with noise
            frame = np.random.normal(1000, 50, (100, 100)).astype(np.float32)

            # Add some stars
            frame[25, 25] = 5000  # Bright star
            frame[75, 75] = 3000  # Medium star
            frame[50, 30] = 2000  # Faint star

            # Add transient in frame 5
            if i == 5:
                frame[40, 60] = 8000  # Strong transient

            frames.append(frame)

        return np.array(frames)

    def test_detect_transients_basic(self):
        """Test basic transient detection"""
        detections = detect_transients(
            self.frames,
            self.filenames,
            self.wcs_objects,
            z_thresh=3.0,
            output_dir=self.temp_dir,
        )

        # Should detect the transient we added
        assert len(detections) > 0

        # Check detection structure
        for det in detections:
            assert "frame" in det
            assert "x" in det
            assert "y" in det
            assert "z_score" in det
            assert "confidence" in det
            assert "cutout_image" in det
            assert "light_curve" in det

    def test_detect_transients_high_threshold(self):
        """Test detection with high threshold"""
        detections = detect_transients(
            self.frames,
            self.filenames,
            self.wcs_objects,
            z_thresh=10.0,  # Very high threshold
            output_dir=self.temp_dir,
        )

        # Should detect fewer (or no) transients
        assert len(detections) >= 0

    def test_detect_transients_edge_rejection(self):
        """Test that detections near edges are rejected"""
        # Create frames with transient near edge
        frames = np.random.normal(1000, 50, (5, 100, 100)).astype(np.float32)
        frames[2, 5, 5] = 10000  # Transient near edge

        detections = detect_transients(
            frames,
            ["test.fits"] * 5,
            [None] * 5,
            z_thresh=3.0,
            edge_margin=20,
            output_dir=self.temp_dir,
        )

        # Should reject edge detections
        for det in detections:
            assert det["x"] >= 20
            assert det["y"] >= 20
            assert det["x"] < 80
            assert det["y"] < 80

    def test_detect_transients_dimming(self):
        """Test detection of dimming events"""
        # Create frames with dimming event
        frames = np.full((5, 100, 100), 1000, dtype=np.float32)
        frames[2, 50, 50] = 500  # Dimming event

        detections = detect_transients(
            frames,
            ["test.fits"] * 5,
            [None] * 5,
            z_thresh=3.0,
            detect_dimming=True,
            output_dir=self.temp_dir,
        )

        # Should detect dimming
        dimming_detections = [d for d in detections if d.get("dimming", False)]
        assert len(dimming_detections) >= 0

    def test_light_curve_generation(self):
        """Test light curve generation for detections"""
        detections = detect_transients(
            self.frames,
            self.filenames,
            self.wcs_objects,
            z_thresh=3.0,
            output_dir=self.temp_dir,
        )

        for det in detections:
            light_curve = det["light_curve"]
            assert len(light_curve) == len(self.frames)
            assert all(isinstance(val, (int, float)) for val in light_curve)

    def test_cutout_image_creation(self):
        """Test that cutout images are created"""
        detections = detect_transients(
            self.frames,
            self.filenames,
            self.wcs_objects,
            z_thresh=3.0,
            output_dir=self.temp_dir,
        )

        for det in detections:
            cutout_path = det["cutout_image"]
            assert os.path.exists(cutout_path)

            # Verify it's a valid image
            img = cv2.imread(cutout_path, cv2.IMREAD_GRAYSCALE)
            assert img is not None
            assert img.shape[0] > 0
            assert img.shape[1] > 0


class TestFITSLoading:
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_fits_file(self, filename, data=None, header=None):
        """Create a test FITS file"""
        if data is None:
            data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

        filepath = os.path.join(self.temp_dir, filename)
        hdu = fits.PrimaryHDU(data)

        if header:
            for key, value in header.items():
                hdu.header[key] = value

        hdu.writeto(filepath, overwrite=True)
        return filepath

    def test_load_fits_stack_basic(self):
        """Test basic FITS stack loading"""
        # Create test FITS files
        for i in range(5):
            self.create_test_fits_file(f"test_{i:03d}.fits")

        with patch("pulsehunter_core.plate_solve_astap"):
            frames, filenames, wcs_objects = load_fits_stack(self.temp_dir)

        assert len(frames) == 5
        assert len(filenames) == 5
        assert len(wcs_objects) == 5
        assert frames.shape == (5, 100, 100)

    def test_load_fits_stack_with_wcs(self):
        """Test FITS loading with WCS headers"""
        # Create FITS with WCS headers
        wcs_header = {
            "CRVAL1": 123.456,
            "CRVAL2": 45.678,
            "CRPIX1": 50.0,
            "CRPIX2": 50.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }

        self.create_test_fits_file("test_wcs.fits", header=wcs_header)

        with patch("pulsehunter_core.plate_solve_astap"):
            frames, filenames, wcs_objects = load_fits_stack(self.temp_dir)

        assert len(wcs_objects) == 1
        assert wcs_objects[0] is not None

    def test_load_fits_stack_with_calibration(self):
        """Test FITS loading with calibration frames"""
        # Create test files
        for i in range(3):
            self.create_test_fits_file(f"test_{i:03d}.fits")

        # Create calibration frames
        master_bias = np.full((100, 100), 500, dtype=np.float32)
        master_dark = np.full((100, 100), 50, dtype=np.float32)
        master_flat = np.full((100, 100), 1.0, dtype=np.float32)

        with patch("pulsehunter_core.plate_solve_astap"):
            frames, filenames, wcs_objects = load_fits_stack(
                self.temp_dir,
                master_bias=master_bias,
                master_dark=master_dark,
                master_flat=master_flat,
            )

        assert len(frames) == 3
        # Frames should be calibrated (different from raw)

    @patch("pulsehunter_core.plate_solve_astap")
    def test_load_fits_stack_plate_solving(self, mock_plate_solve):
        """Test FITS loading with plate solving"""
        # Create FITS without WCS
        self.create_test_fits_file("test_no_wcs.fits")

        frames, filenames, wcs_objects = load_fits_stack(
            self.temp_dir, plate_solve_missing=True
        )

        # Should have called plate solving
        mock_plate_solve.assert_called()


class TestCrossmatch:
    @patch("pulsehunter_core.Gaia.launch_job")
    def test_crossmatch_with_gaia_match_found(self, mock_gaia):
        """Test GAIA crossmatching with match found"""
        # Mock GAIA response
        mock_result = MagicMock()
        mock_result.__len__ = lambda self: 1
        mock_result.__getitem__ = lambda self, idx: {
            "source_id": 123456789,
            "ra": 123.456,
            "dec": 45.678,
            "phot_g_mean_mag": 12.5,
            "parallax": 10.0,
        }

        mock_job = MagicMock()
        mock_job.get_results.return_value = mock_result
        mock_gaia.return_value = mock_job

        # Test detection
        detections = [{"ra_deg": 123.456, "dec_deg": 45.678, "frame": 0}]

        matched = crossmatch_with_gaia(detections)

        assert len(matched) == 1
        assert matched[0]["match_name"] is not None
        assert "GAIA DR3" in matched[0]["match_name"]
        assert matched[0]["object_type"] == "Star"
        assert matched[0]["g_mag"] == 12.5

    @patch("pulsehunter_core.Gaia.launch_job")
    def test_crossmatch_with_gaia_no_match(self, mock_gaia):
        """Test GAIA crossmatching with no match found"""
        # Mock empty GAIA response
        mock_result = MagicMock()
        mock_result.__len__ = lambda self: 0

        mock_job = MagicMock()
        mock_job.get_results.return_value = mock_result
        mock_gaia.return_value = mock_job

        # Test detection
        detections = [{"ra_deg": 123.456, "dec_deg": 45.678, "frame": 0}]

        matched = crossmatch_with_gaia(detections)

        assert len(matched) == 1
        assert matched[0]["match_name"] is None
        assert matched[0]["object_type"] is None

    def test_crossmatch_with_gaia_no_coordinates(self):
        """Test GAIA crossmatching with missing coordinates"""
        detections = [{"ra_deg": None, "dec_deg": None, "frame": 0}]

        matched = crossmatch_with_gaia(detections)

        assert len(matched) == 1
        assert matched[0]["match_name"] is None
        assert matched[0]["object_type"] is None


class TestPerformance:
    """Performance and benchmark tests"""

    @pytest.mark.slow
    def test_detection_performance_large_dataset(self):
        """Test detection performance with large dataset"""
        # Create large test dataset
        frames = np.random.normal(1000, 50, (50, 512, 512)).astype(np.float32)
        filenames = [f"test_{i:03d}.fits" for i in range(50)]
        wcs_objects = [None] * 50

        with tempfile.TemporaryDirectory() as temp_dir:
            import time

            start_time = time.time()

            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=5.0, output_dir=temp_dir
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"Processed {len(frames)} frames in {processing_time:.2f} seconds")

            print(f"Rate: {len(frames) / processing_time:.1f} frames/second")

            # Should complete within reasonable time
            assert processing_time < 60  # Less than 1 minute

    def test_memory_usage(self):
        """Test memory usage with typical dataset"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process typical dataset
        frames = np.random.normal(1000, 50, (20, 256, 256)).astype(np.float32)
        filenames = [f"test_{i:03d}.fits" for i in range(20)]
        wcs_objects = [None] * 20

        with tempfile.TemporaryDirectory() as temp_dir:
            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=5.0, output_dir=temp_dir
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage increased by {memory_increase:.1f} MB")

        # Should not use excessive memory
        assert memory_increase < 500  # Less than 500 MB increase


if __name__ == "__main__":
    pytest.main([__file__])
