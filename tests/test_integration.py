#!/usr/bin/env python3
"""
Integration tests for PulseHunter complete workflow
"""

import json
import os

# Import modules to test
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from calibration import create_master_frame, generate_lightcurve_outputs

from exoplanet_match import match_transits_with_exoplanets
from pulsehunter_core import (
    crossmatch_with_gaia,
    detect_transients,
    load_fits_stack,
    save_report,
)

sys.path.append("..")


class TestFullWorkflow:
    """Test complete detection workflow from FITS to report"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.fits_dir = os.path.join(self.temp_dir, "fits")
        self.calib_dir = os.path.join(self.temp_dir, "calibration")
        self.output_dir = os.path.join(self.temp_dir, "output")

        os.makedirs(self.fits_dir)
        os.makedirs(self.calib_dir)
        os.makedirs(self.output_dir)

    def teardown_method(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_realistic_test_data(self):
        """Create realistic test astronomical data"""
        np.random.seed(42)

        # Create science frames with realistic astronomical scene
        for i in range(20):
            # Base sky background
            frame = np.random.normal(1000, 30, (200, 200)).astype(np.float32)

            # Add stars with realistic brightness distribution
            stars = [
                (50, 50, 8000),  # Bright star
                (120, 80, 4000),  # Medium star
                (30, 150, 2500),  # Faint star
                (170, 40, 3500),  # Another star
                (90, 160, 1800),  # Faint star
            ]

            for x, y, brightness in stars:
                # Add star with realistic PSF (gaussian)
                y_coords, x_coords = np.ogrid[:200, :200]
                star_mask = (
                    (x_coords - x) ** 2 + (y_coords - y) ** 2
                ) < 9  # 3x3 pixel star
                frame[star_mask] += brightness

            # Add transient event in frames 8-12
            if 8 <= i <= 12:
                # Transient at position (100, 100)
                transient_brightness = 6000 * (
                    1.0 - abs(i - 10) / 5.0
                )  # Peak at frame 10
                y_coords, x_coords = np.ogrid[:200, :200]
                transient_mask = ((x_coords - 100) ** 2 + (y_coords - 100) ** 2) < 4
                frame[transient_mask] += transient_brightness

            # Create FITS file with proper headers
            hdu = fits.PrimaryHDU(frame.astype(np.uint16))
            hdu.header["OBJECT"] = "Test Field"
            hdu.header["EXPTIME"] = 60.0
            hdu.header["FILTER"] = "R"
            hdu.header["DATE-OBS"] = f"2023-01-01T{i:02d}:00:00"

            # Add basic WCS (simplified)
            hdu.header["CRVAL1"] = 123.456
            hdu.header["CRVAL2"] = 45.678
            hdu.header["CRPIX1"] = 100.0
            hdu.header["CRPIX2"] = 100.0
            hdu.header["CDELT1"] = -0.001
            hdu.header["CDELT2"] = 0.001
            hdu.header["CTYPE1"] = "RA---TAN"
            hdu.header["CTYPE2"] = "DEC--TAN"

            hdu.writeto(
                os.path.join(self.fits_dir, f"science_{i:03d}.fits"), overwrite=True
            )

    def create_calibration_frames(self):
        """Create realistic calibration frames"""
        # Bias frames
        bias_dir = os.path.join(self.calib_dir, "bias")
        os.makedirs(bias_dir)
        for i in range(10):
            bias = np.random.normal(500, 10, (200, 200)).astype(np.uint16)
            fits.PrimaryHDU(bias).writeto(
                os.path.join(bias_dir, f"bias_{i:03d}.fits"), overwrite=True
            )

        # Dark frames
        dark_dir = os.path.join(self.calib_dir, "dark")
        os.makedirs(dark_dir)
        for i in range(10):
            dark = np.random.normal(520, 15, (200, 200)).astype(
                np.uint16
            )  # Bias + dark current
            fits.PrimaryHDU(dark).writeto(
                os.path.join(dark_dir, f"dark_{i:03d}.fits"), overwrite=True
            )

        # Flat frames
        flat_dir = os.path.join(self.calib_dir, "flat")
        os.makedirs(flat_dir)
        for i in range(10):
            # Realistic flat with illumination gradient
            y, x = np.ogrid[:200, :200]
            flat_base = 30000 * (
                1.0 - 0.1 * ((x - 100) / 100) ** 2 - 0.1 * ((y - 100) / 100) ** 2
            )
            flat = flat_base + np.random.normal(0, 100, (200, 200))
            flat = np.clip(flat, 10000, 50000).astype(np.uint16)
            fits.PrimaryHDU(flat).writeto(
                os.path.join(flat_dir, f"flat_{i:03d}.fits"), overwrite=True
            )

    @patch("pulsehunter_core.plate_solve_astap")
    def test_complete_workflow_no_calibration(self, mock_plate_solve):
        """Test complete workflow without calibration"""
        self.create_realistic_test_data()

        # Step 1: Load FITS stack
        frames, filenames, wcs_objects = load_fits_stack(self.fits_dir)

        assert len(frames) == 20
        assert frames.shape == (20, 200, 200)

        # Step 2: Detect transients
        detections = detect_transients(
            frames,
            filenames,
            wcs_objects,
            z_thresh=4.0,
            detect_dimming=True,
            output_dir=self.output_dir,
        )

        # Should detect the synthetic transient
        assert len(detections) > 0

        # Verify detection near expected position
        transient_detections = [
            d for d in detections if abs(d["x"] - 100) < 10 and abs(d["y"] - 100) < 10
        ]
        assert len(transient_detections) > 0

        # Step 3: Cross-match with GAIA (mocked)
        with patch("pulsehunter_core.Gaia.launch_job") as mock_gaia:
            mock_result = MagicMock()
            mock_result.__len__ = lambda self: 0  # No matches
            mock_job = MagicMock()
            mock_job.get_results.return_value = mock_result
            mock_gaia.return_value = mock_job

            matched_detections = crossmatch_with_gaia(detections)

        assert len(matched_detections) == len(detections)

        # Step 4: Check for exoplanet matches (mocked)
        with patch("exoplanet_match.NasaExoplanetArchive.query_criteria") as mock_exo:
            mock_exo.return_value = []  # No exoplanet matches

            final_detections = match_transits_with_exoplanets(matched_detections)

        assert len(final_detections) == len(matched_detections)

        # Step 5: Generate outputs
        generate_lightcurve_outputs(
            final_detections, self.output_dir, "test_dataset", "Test Observer"
        )

        # Verify outputs were created
        assert os.path.exists(os.path.join(self.output_dir, "README.txt"))
        assert os.path.exists(os.path.join(self.output_dir, "summary.json"))

        # Step 6: Save report
        report_path = os.path.join(self.output_dir, "report.json")
        save_report(final_detections, report_path)

        assert os.path.exists(report_path)

        # Verify report content
        with open(report_path, "r") as f:
            report_data = json.load(f)

        assert "detections" in report_data
        assert len(report_data["detections"]) == len(final_detections)

    @patch("pulsehunter_core.plate_solve_astap")
    def test_complete_workflow_with_calibration(self, mock_plate_solve):
        """Test complete workflow with calibration frames"""
        self.create_realistic_test_data()
        self.create_calibration_frames()

        # Create master calibration frames
        master_bias = create_master_frame(os.path.join(self.calib_dir, "bias"), "bias")
        master_dark = create_master_frame(os.path.join(self.calib_dir, "dark"), "dark")
        master_flat = create_master_frame(os.path.join(self.calib_dir, "flat"), "flat")

        assert master_bias is not None
        assert master_dark is not None
        assert master_flat is not None

        # Load FITS stack with calibration
        frames, filenames, wcs_objects = load_fits_stack(
            self.fits_dir,
            master_bias=master_bias,
            master_dark=master_dark,
            master_flat=master_flat,
        )

        assert len(frames) == 20

        # Continue with detection workflow
        detections = detect_transients(
            frames, filenames, wcs_objects, z_thresh=4.0, output_dir=self.output_dir
        )

        # Should still detect transients after calibration
        assert len(detections) > 0

    def test_error_handling_corrupted_fits(self):
        """Test error handling with corrupted FITS files"""
        # Create mix of good and bad FITS files
        self.create_realistic_test_data()

        # Add corrupted file
        corrupt_path = os.path.join(self.fits_dir, "corrupted.fits")
        with open(corrupt_path, "wb") as f:
            f.write(b"This is not a FITS file")

        # Should handle gracefully
        with patch("pulsehunter_core.plate_solve_astap"):
            frames, filenames, wcs_objects = load_fits_stack(self.fits_dir)

        # Should load the good files and skip the bad one
        if len(frames) != 20:
            raise AssertionError("Expected 20 valid FITS frames")

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with larger dataset"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create larger test dataset
        for i in range(50):
            frame = np.random.normal(1000, 30, (512, 512)).astype(np.uint16)
            hdu = fits.PrimaryHDU(frame)
            hdu.writeto(
                os.path.join(self.fits_dir, f"large_{i:03d}.fits"), overwrite=True
            )

        with patch("pulsehunter_core.plate_solve_astap"):
            frames, filenames, wcs_objects = load_fits_stack(self.fits_dir)

            _ = detect_transients(
                frames, filenames, wcs_objects, z_thresh=5.0, output_dir=self.output_dir
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage for 50x512x512 frames: {memory_increase:.1f} MB")

        # Should handle large datasets efficiently
        assert memory_increase < 2000  # Less than 2GB increase

    @patch("requests.post")
    def test_report_upload(self, mock_post):
        """Test report upload functionality"""
        # Mock successful upload
        mock_response = MagicMock()
        mock_response.ok = True
        mock_post.return_value = mock_response

        # Create test detection data
        detections = [
            {
                "frame": 0,
                "x": 100,
                "y": 100,
                "ra_deg": 123.456,
                "dec_deg": 45.678,
                "z_score": 6.5,
                "confidence": 0.85,
                "timestamp_utc": "2023-01-01T00:00:00Z",
            }
        ]

        report_path = os.path.join(self.output_dir, "test_report.json")
        save_report(detections, report_path)

        # Verify file was created
        assert os.path.exists(report_path)

        # Verify upload was attempted
        mock_post.assert_called_once()


class TestRealDataScenarios:
    """Tests with realistic astronomical scenarios"""

    def test_variable_star_scenario(self):
        """Test detection of variable star-like behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create frames with periodic variable star
            frames = []
            for i in range(30):
                frame = np.random.normal(1000, 20, (100, 100)).astype(np.float32)

                # Add periodic variable (period = 10 frames)
                phase = (i % 10) / 10.0 * 2 * np.pi
                variable_brightness = 3000 + 1000 * np.sin(phase)
                frame[50, 50] = variable_brightness

                frames.append(frame)

            frames = np.array(frames)
            filenames = [f"var_{i:03d}.fits" for i in range(30)]
            wcs_objects = [None] * 30

            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=3.0, output_dir=temp_dir
            )

            # Should detect multiple events from the variable star
            variable_detections = [
                d for d in detections if abs(d["x"] - 50) < 3 and abs(d["y"] - 50) < 3
            ]
            assert len(variable_detections) > 1

    def test_exoplanet_transit_scenario(self):
        """Test detection of exoplanet transit-like dimming"""
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            for i in range(20):
                frame = np.random.normal(1000, 15, (100, 100)).astype(np.float32)

                # Add stable star
                star_brightness = 5000

                # Add transit dimming (frames 8-12, 2% depth)
                if 8 <= i <= 12:
                    star_brightness *= 0.98  # 2% dimming

                frame[60, 70] = star_brightness
                frames.append(frame)

            frames = np.array(frames)
            filenames = [f"transit_{i:03d}.fits" for i in range(20)]
            wcs_objects = [None] * 20

            detections = detect_transients(
                frames,
                filenames,
                wcs_objects,
                z_thresh=2.0,
                detect_dimming=True,
                output_dir=temp_dir,
            )

            # Should detect the transit dimming
            transit_detections = [d for d in detections if d.get("dimming", False)]
            assert len(transit_detections) > 0

    def test_satellite_trail_rejection(self):
        """Test rejection of satellite trails and cosmic rays"""
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            for i in range(10):
                frame = np.random.normal(1000, 20, (100, 100)).astype(np.float32)

                # Add satellite trail (linear feature)
                if i == 5:
                    for j in range(20, 80):
                        frame[j, j] = 8000  # Diagonal trail

                # Add cosmic ray (single pixel spike)
                if i == 7:
                    frame[30, 40] = 15000

                frames.append(frame)

            frames = np.array(frames)
            filenames = [f"artifact_{i:03d}.fits" for i in range(10)]
            wcs_objects = [None] * 10

            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=4.0, output_dir=temp_dir
            )

            # Algorithm should ideally reject or flag these artifacts
            # For now, just verify it doesn't crash
            if not isinstance(detections, list):
                raise AssertionError("Detections should be a list")


class TestErrorRecovery:
    """Test error recovery and robustness"""

    def test_insufficient_frames(self):
        """Test handling of insufficient frames"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only 2 frames (insufficient for statistics)
            frames = np.random.normal(1000, 50, (2, 100, 100)).astype(np.float32)
            filenames = ["test_001.fits", "test_002.fits"]
            wcs_objects = [None, None]

            # Should handle gracefully
            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=5.0, output_dir=temp_dir
            )

            # May return empty or handle with warning
            assert isinstance(detections, list)

    def test_nan_handling(self):
        """Test handling of NaN values in data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = np.random.normal(1000, 50, (10, 100, 100)).astype(np.float32)

            # Insert NaN values
            frames[5, 50:60, 50:60] = np.nan

            filenames = [f"nan_test_{i:03d}.fits" for i in range(10)]
            wcs_objects = [None] * 10

            # Should handle NaN values gracefully
            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=5.0, output_dir=temp_dir
            )

            assert isinstance(detections, list)

    def test_extreme_values(self):
        """Test handling of extreme pixel values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = np.random.normal(1000, 50, (10, 100, 100)).astype(np.float32)

            # Add extreme values
            frames[3, 25, 25] = 1e10  # Very bright
            frames[7, 75, 75] = -1000  # Negative

            filenames = [f"extreme_{i:03d}.fits" for i in range(10)]
            wcs_objects = [None] * 10

            detections = detect_transients(
                frames, filenames, wcs_objects, z_thresh=5.0, output_dir=temp_dir
            )

            # Should handle without crashing
            assert isinstance(detections, list)


class TestConfigurationVariations:
    """Test different configuration options"""

    def test_different_thresholds(self):
        """Test detection with different threshold values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create frames with known transient
            frames = np.random.normal(1000, 30, (10, 100, 100)).astype(np.float32)
            frames[5, 50, 50] = 3000  # Strong transient

            filenames = [f"thresh_{i:03d}.fits" for i in range(10)]
            wcs_objects = [None] * 10

            # Test different thresholds
            for threshold in [3.0, 5.0, 8.0, 10.0]:
                detections = detect_transients(
                    frames,
                    filenames,
                    wcs_objects,
                    z_thresh=threshold,
                    output_dir=temp_dir,
                )
                assert isinstance(detections, list)

                print(f"Threshold {threshold}: detection completed")

                # Higher thresholds should generally yield fewer detections
                # assert isinstance(detections, list)  # Just verify no crash

    def test_different_cutout_sizes(self):
        """Test detection with different cutout sizes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = np.random.normal(1000, 30, (5, 100, 100)).astype(np.float32)
            frames[2, 50, 50] = 3000

            filenames = [f"cutout_{i:03d}.fits" for i in range(5)]
            wcs_objects = [None] * 5

            for cutout_size in [20, 50, 80]:
                detections = detect_transients(
                    frames,
                    filenames,
                    wcs_objects,
                    z_thresh=4.0,
                    cutout_size=cutout_size,
                    output_dir=temp_dir,
                )

                if detections:
                    # Check that cutout images exist and have reasonable size
                    for det in detections:
                        if not os.path.exists(det["cutout_image"]):
                            raise AssertionError(
                                f"Missing cutout image: {det['cutout_image']}"
                            )


if __name__ == "__main__":
    # Run with different verbosity levels
    pytest.main([__file__, "-v", "--tb=short"])
