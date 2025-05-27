#!/usr/bin/env python3
"""
Test suite for PulseHunter calibration module
"""

import os

# Import the modules to test
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from calibration import (
    create_master_frame,
    generate_dataset_id,
    generate_lightcurve_outputs,
)

sys.path.append("..")


class TestCalibration:
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_fits(self, filename, data=None):
        """Create a test FITS file"""
        if data is None:
            data = self.test_data
        filepath = os.path.join(self.temp_dir, filename)
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(filepath, overwrite=True)
        return filepath

    def test_create_master_frame_single_file(self):
        """Test master frame creation with single file"""
        self.create_test_fits("test_dark_001.fits")

        master = create_master_frame(self.temp_dir, "dark")

        assert master is not None
        assert master.shape == self.test_data.shape
        assert master.dtype == np.float32

    def test_create_master_frame_multiple_files(self):
        """Test master frame creation with multiple files"""
        # Create multiple test files
        for i in range(5):
            data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
            self.create_test_fits(f"test_dark_{i:03d}.fits", data)

        master = create_master_frame(self.temp_dir, "dark")

        assert master is not None
        assert master.shape == (100, 100)
        assert master.dtype == np.float32

    def test_create_master_frame_no_files(self):
        """Test master frame creation with no FITS files"""
        with pytest.raises(ValueError, match="No usable calibration frames found"):
            create_master_frame(self.temp_dir, "dark")

    def test_create_master_frame_corrupted_file(self):
        """Test master frame creation with corrupted file"""
        # Create a corrupted FITS file
        filepath = os.path.join(self.temp_dir, "corrupted.fits")
        with open(filepath, "wb") as f:
            f.write(b"not a fits file")

        # Should handle the error gracefully
        with pytest.raises(ValueError, match="No usable calibration frames found"):
            create_master_frame(self.temp_dir, "dark")

    def test_generate_dataset_id(self):
        """Test dataset ID generation"""
        # Create some test FITS files
        self.create_test_fits("test_001.fits")
        self.create_test_fits("test_002.fits")

        dataset_id = generate_dataset_id(self.temp_dir)

        assert dataset_id is not None
        assert len(dataset_id) == 64  # SHA256 hash length
        assert isinstance(dataset_id, str)

        # Test reproducibility
        dataset_id2 = generate_dataset_id(self.temp_dir)
        assert dataset_id == dataset_id2

    def test_generate_dataset_id_no_fits(self):
        """Test dataset ID generation with no FITS files"""
        dataset_id = generate_dataset_id(self.temp_dir)
        assert dataset_id is not None  # Should still generate an ID

    def test_generate_lightcurve_outputs(self):
        """Test light curve output generation"""
        # Mock detection data
        detections = [
            {
                "light_curve": [100, 95, 90, 95, 100, 105, 100],
                "ra_deg": 123.456,
                "dec_deg": 45.678,
                "observer": "Test Observer",
                "timestamp_utc": "2023-01-01T00:00:00Z",
                "confidence": 0.85,
                "match_name": "Test Star",
                "g_mag": 12.5,
            }
        ]

        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Mock matplotlib to avoid display issues in tests
        with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.plot"), patch(
            "matplotlib.pyplot.title"
        ), patch("matplotlib.pyplot.xlabel"), patch("matplotlib.pyplot.ylabel"), patch(
            "matplotlib.pyplot.grid"
        ), patch(
            "matplotlib.pyplot.figtext"
        ), patch(
            "matplotlib.pyplot.tight_layout"
        ), patch(
            "matplotlib.pyplot.savefig"
        ), patch(
            "matplotlib.pyplot.close"
        ):
            generate_lightcurve_outputs(
                detections, output_dir, "test_dataset", "Test Observer"
            )

        # Check that CSV file was created
        csv_file = os.path.join(output_dir, "lightcurve_0000.csv")
        assert os.path.exists(csv_file)

        # Check CSV content
        with open(csv_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 8  # Header + 7 data points
            assert "Frame,Brightness" in lines[0]

        # Check that README was created
        readme_file = os.path.join(output_dir, "README.txt")
        assert os.path.exists(readme_file)

        # Check that summary JSON was created
        summary_file = os.path.join(output_dir, "summary.json")
        assert os.path.exists(summary_file)


class TestCalibrationIntegration:
    """Integration tests for calibration workflow"""

    def test_full_calibration_workflow(self):
        """Test complete calibration workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test calibration frames
            bias_dir = os.path.join(temp_dir, "bias")
            dark_dir = os.path.join(temp_dir, "dark")
            flat_dir = os.path.join(temp_dir, "flat")

            os.makedirs(bias_dir)
            os.makedirs(dark_dir)
            os.makedirs(flat_dir)

            # Create test frames
            for i in range(3):
                # Bias frames (low values)
                bias_data = np.random.randint(500, 600, (50, 50), dtype=np.uint16)
                fits.PrimaryHDU(bias_data).writeto(
                    os.path.join(bias_dir, f"bias_{i:03d}.fits"), overwrite=True
                )

                # Dark frames (bias + dark current)
                dark_data = np.random.randint(550, 650, (50, 50), dtype=np.uint16)
                fits.PrimaryHDU(dark_data).writeto(
                    os.path.join(dark_dir, f"dark_{i:03d}.fits"), overwrite=True
                )

                # Flat frames (normalized)
                flat_data = np.random.randint(20000, 40000, (50, 50), dtype=np.uint16)
                fits.PrimaryHDU(flat_data).writeto(
                    os.path.join(flat_dir, f"flat_{i:03d}.fits"), overwrite=True
                )

            # Create master frames
            master_bias = create_master_frame(bias_dir, "bias")
            master_dark = create_master_frame(dark_dir, "dark")
            master_flat = create_master_frame(flat_dir, "flat")

            # Verify masters were created
            assert master_bias is not None
            assert master_dark is not None
            assert master_flat is not None

            # Verify dimensions
            assert master_bias.shape == (50, 50)
            assert master_dark.shape == (50, 50)
            assert master_flat.shape == (50, 50)

            # Verify data types
            assert master_bias.dtype == np.float32
            assert master_dark.dtype == np.float32
            assert master_flat.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__])
