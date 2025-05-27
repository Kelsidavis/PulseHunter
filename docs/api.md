# PulseHunter API Reference

This document provides detailed information about PulseHunter's programming interface for developers and advanced users.

## Core Modules

### pulsehunter_core

The main detection and processing engine.

#### Functions

##### `load_fits_stack(folder, **kwargs)`

Load and calibrate a sequence of FITS images.

**Parameters:**
- `folder` (str): Path to directory containing FITS files
- `plate_solve_missing` (bool, optional): Enable plate solving for missing WCS. Default: False
- `astap_exe` (str, optional): Path to ASTAP executable. Default: "astap"
- `master_bias` (np.ndarray, optional): Master bias frame for calibration
- `master_dark` (np.ndarray, optional): Master dark frame for calibration
- `master_flat` (np.ndarray, optional): Master flat frame for calibration
- `camera_mode` (str, optional): Camera type ("mono" or "osc"). Default: "mono"
- `filter_name` (str, optional): Filter name for metadata

**Returns:**
- `tuple`: (frames, filenames, wcs_objects)
  - `frames` (np.ndarray): 3D array of calibrated images [frame, y, x]
  - `filenames` (list): List of FITS filenames
  - `wcs_objects` (list): World Coordinate System objects for each frame

**Example:**
```python
frames, filenames, wcs_list = load_fits_stack(
    "/path/to/fits",
    plate_solve_missing=True,
    master_bias=bias_frame,
    camera_mode="mono"
)
```

##### `detect_transients(frames, filenames, wcs_objects, **kwargs)`

Detect transient events in astronomical image sequence.

**Parameters:**
- `frames` (np.ndarray): 3D array of calibrated images [frame, y, x]
- `filenames` (list): List of corresponding FITS filenames
- `wcs_objects` (list): WCS objects for coordinate transformation
- `output_dir` (str, optional): Directory for output files. Default: "detections"
- `z_thresh` (float, optional): Detection threshold in standard deviations. Default: 6.0
- `cutout_size` (int, optional): Size of detection cutouts in pixels. Default: 50
- `edge_margin` (int, optional): Margin from image edges in pixels. Default: 20
- `detect_dimming` (bool, optional): Enable detection of dimming events. Default: False

**Returns:**
- `list`: List of detection dictionaries containing:
  - `frame` (int): Frame number of detection
  - `filename` (str): FITS filename
  - `x`, `y` (int): Pixel coordinates
  - `ra_deg`, `dec_deg` (float): Sky coordinates (if WCS available)
  - `z_score` (float): Statistical significance
  - `dimming` (bool): True for dimming events
  - `confidence` (float): Detection confidence (0-1)
  - `cutout_image` (str): Path to cutout image
  - `timestamp_utc` (str): UTC timestamp
  - `light_curve` (list): Brightness measurements across all frames

**Example:**
```python
detections = detect_transients(
    frames, filenames, wcs_objects,
    z_thresh=5.0,
    detect_dimming=True,
    output_dir="./results"
)
```

##### `crossmatch_with_gaia(detections, radius_arcsec=5.0)`

Cross-match detections with GAIA DR3 catalog.

**Parameters:**
- `detections` (list): List of detection dictionaries
- `radius_arcsec` (float, optional): Search radius in arcseconds. Default: 5.0

**Returns:**
- `list`: Updated detections with GAIA information:
  - `match_name` (str): GAIA source identifier
  - `object_type` (str): Object classification
  - `angular_distance_arcsec` (float): Separation from catalog position
  - `g_mag` (float): GAIA G magnitude
  - `distance_pc` (float): Distance in parsecs (if parallax available)

**Example:**
```python
matched_detections = crossmatch_with_gaia(detections, radius_arcsec=3.0)
```

##### `plate_solve_astap(filepath, astap_exe="astap")`

Perform astrometric calibration using ASTAP.

**Parameters:**
- `filepath` (str): Path to FITS file
- `astap_exe` (str, optional): ASTAP executable path. Default: "astap"

**Returns:**
- None (modifies FITS file in place)

**Example:**
```python
plate_solve_astap("/path/to/image.fits", astap_exe="/usr/local/bin/astap")
```

##### `save_report(detections, output_path="pulse_report.json")`

Save detection report and optionally upload to network.

**Parameters:**
- `detections` (list): List of detection dictionaries
- `output_path` (str, optional): Output file path. Default: "pulse_report.json"

**Returns:**
- None

**Example:**
```python
save_report(detections, "my_observations_2023.json")
```

### calibration

Calibration frame processing and observatory setup.

#### Functions

##### `create_master_frame(folder, kind="dark")`

Create master calibration frame from directory of FITS files.

**Parameters:**
- `folder` (str): Directory containing calibration frames
- `kind` (str, optional): Frame type ("bias", "dark", "flat"). Default: "dark"

**Returns:**
- `np.ndarray`: Master calibration frame (float32)

**Raises:**
- `ValueError`: If no usable frames found

**Example:**
```python
master_dark = create_master_frame("/path/to/darks", "dark")
master_flat = create_master_frame("/path/to/flats", "flat")
```

##### `generate_dataset_id(folder)`

Generate unique identifier for dataset.

**Parameters:**
- `folder` (str): Path to FITS directory

**Returns:**
- `str`: SHA256 hash of folder path and file list

**Example:**
```python
dataset_id = generate_dataset_id("/path/to/observations")
print(f"Dataset ID: {dataset_id[:16]}...")
```

##### `generate_lightcurve_outputs(detections, output_folder, dataset_id, observer)`

Generate comprehensive light curve analysis outputs.

**Parameters:**
- `detections` (list): Detection dictionaries with light_curve data
- `output_folder` (str): Output directory path
- `dataset_id` (str): Unique dataset identifier
- `observer` (str): Observer name

**Returns:**
- None (creates files in output directory)

**Creates:**
- `lightcurve_XXXX.csv`: Photometry data for each detection
- `lightcurve_XXXX.png`: Light curve plots
- `README.txt`: Human-readable summary
- `summary.json`: Machine-readable metadata

**Example:**
```python
generate_lightcurve_outputs(
    detections,
    "./analysis_output",
    "dataset_20231201",
    "Observer Name"
)
```

##### `open_calibration_dialog()`

Interactive calibration setup dialog (GUI).

**Parameters:**
- None

**Returns:**
- `dict` or `None`: Calibration configuration dictionary containing:
  - `observer` (str): Observer name
  - `astap` (str): ASTAP executable path
  - `dataset_id` (str): Generated dataset ID
  - `dataset_folder` (str): Selected FITS folder
  - `bias`, `dark`, `flat` (np.ndarray): Master calibration frames
  - `camera_mode` (str): Camera type

**Example:**
```python
config = open_calibration_dialog()
if config:
    print(f"Observer: {config['observer']}")
    print(f"Camera: {config['camera_mode']}")
```

### exoplanet_match

Exoplanet transit candidate identification.

#### Functions

##### `match_transits_with_exoplanets(detections, radius_arcsec=5.0)`

Cross-match dimming events with known exoplanets.

**Parameters:**
- `detections` (list): Detection dictionaries
- `radius_arcsec` (float, optional): Search radius. Default: 5.0

**Returns:**
- `list`: Updated detections with exoplanet information:
  - `exo_match` (dict): Exoplanet match information
    - `host` (str): Host star name
    - `planet` (str): Planet designation
    - `sep_arcsec` (float): Angular separation
    - `period_days` (float): Orbital period
    - `depth_ppm` (float): Transit depth in parts per million

**Example:**
```python
from exoplanet_match import match_transits_with_exoplanets

# Filter for dimming events only
dimming_events = [d for d in detections if d.get('dimming', False)]

# Check for exoplanet matches
exo_candidates = match_transits_with_exoplanets(dimming_events)

# Report matches
for det in exo_candidates:
    if det.get('exo_match'):
        match = det['exo_match']
        print(f"Potential transit: {match['planet']} around {match['host']}")
```

## Data Structures

### Detection Dictionary

Standard format for detection objects:

```python
detection = {
    # Basic detection info
    "frame": 42,                    # Frame number (int)
    "filename": "obs_042.fits",     # Source filename (str)
    "x": 256,                       # X pixel coordinate (int)
    "y": 128,                       # Y pixel coordinate (int)

    # Astrometry (if available)
    "ra_deg": 123.456789,          # Right ascension (float)
    "dec_deg": 45.678901,          # Declination (float)

    # Detection metrics
    "z_score": 6.85,               # Statistical significance (float)
    "dimming": False,              # Dimming event flag (bool)
    "confidence": 0.92,            # Confidence score 0-1 (float)

    # Output files
    "cutout_image": "det_042.png", # Path to cutout image (str)
    "timestamp_utc": "2023-12-01T23:45:30Z", # ISO timestamp (str)
    "light_curve": [1000, 1020, 3500, 1015, 995], # Brightness values (list)

    # Catalog matches (added by crossmatching)
    "match_name": "GAIA DR3 123456789",        # Catalog identifier (str/None)
    "object_type": "Star",                     # Object classification (str/None)
    "angular_distance_arcsec": 2.3,            # Match separation (float/None)
    "g_mag": 12.45,                           # GAIA G magnitude (float/None)
    "distance_pc": 156.7,                     # Distance in parsecs (float/None)

    # Exoplanet matches (added by exoplanet matching)
    "exo_match": {                            # Exoplanet match info (dict/None)
        "host": "HD 189733",
        "planet": "HD 189733 b",
        "sep_arcsec": 1.8,
        "period_days": 2.218575,
        "depth_ppm": 2550.0
    }
}
```

### Configuration Dictionary

Calibration and observatory configuration:

```python
config = {
    "observer": "Jane Astronomer",             # Observer name (str)
    "astap": "/usr/local/bin/astap",          # ASTAP path (str)
    "dataset_id": "a1b2c3d4...",             # Dataset hash (str)
    "dataset_folder": "/path/to/fits",        # FITS directory (str)
    "bias": np.ndarray(...),                  # Master bias (ndarray/None)
    "dark": np.ndarray(...),                  # Master dark (ndarray/None)
    "flat": np.ndarray(...),                  # Master flat (ndarray/None)
    "camera_mode": "mono"                     # Camera type (str)
}
```

## Error Handling

### Common Exceptions

```python
try:
    frames, filenames, wcs_objects = load_fits_stack(folder)
except FileNotFoundError:
    print("FITS directory not found")
except ValueError as e:
    print(f"Invalid FITS data: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

try:
    detections = detect_transients(frames, filenames, wcs_objects)
except MemoryError:
    print("Insufficient memory for dataset")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

### Validation Functions

```python
def validate_detection(detection):
    """Validate detection dictionary structure"""
    required_fields = ['frame', 'x', 'y', 'z_score', 'confidence']
    for field in required_fields:
        if field not in detection:
            raise ValueError(f"Missing required field: {field}")

    if not (0 <= detection['confidence'] <= 1):
        raise ValueError("Confidence must be between 0 and 1")

    return True

def validate_fits_directory(folder):
    """Check if directory contains valid FITS files"""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")

    fits_files = [f for f in os.listdir(folder) if f.endswith('.fits')]
    if len(fits_files) < 10:
        raise ValueError(f"Insufficient FITS files: {len(fits_files)} (minimum 10)")

    return fits_files
```

## Performance Considerations

### Memory Management

```python
# Process large datasets in batches
def process_large_dataset(folder, batch_size=50):
    fits_files = sorted([f for f in os.listdir(folder) if f.endswith('.fits')])

    all_detections = []
    for i in range(0, len(fits_files), batch_size):
        batch_files = fits_files[i:i+batch_size]

        # Create temporary directory for batch
        with tempfile.TemporaryDirectory() as temp_dir:
            # Symlink batch files
            for f in batch_files:
                src = os.path.join(folder, f)
                dst = os.path.join(temp_dir, f)
                os.symlink(src, dst)

            # Process batch
            frames, filenames, wcs_objects = load_fits_stack(temp_dir)
            detections = detect_transients(frames, filenames, wcs_objects)
            all_detections.extend(detections)

            # Clean up memory
            del frames
            gc.collect()

    return all_detections
```

### Parallel Processing

```python
from multiprocessing import Pool
import functools

def process_detection_batch(detection_batch, output_dir):
    """Process a batch of detections in parallel"""
    # Implementation for batch processing
    pass

def parallel_crossmatch(detections, max_workers=4):
    """Perform GAIA crossmatching in parallel"""
    batch_size = len(detections) // max_workers
    batches = [detections[i:i+batch_size] for i in range(0, len(detections), batch_size)]

    with Pool(max_workers) as pool:
        results = pool.map(crossmatch_with_gaia, batches)

    # Combine results
    return [det for batch in results for det in batch]
```

## Integration Examples

### Command Line Interface

```python
#!/usr/bin/env python3
"""Command line interface for PulseHunter"""

import argparse
from pulsehunter_core import load_fits_stack, detect_transients, crossmatch_with_gaia

def main():
    parser = argparse.ArgumentParser(description='PulseHunter Detection Pipeline')
    parser.add_argument('input_dir', help='Directory containing FITS files')
    parser.add_argument('--threshold', type=float, default=6.0, help='Detection threshold')
    parser.add_argument('--output', default='detections', help='Output directory')
    parser.add_argument('--dimming', action='store_true', help='Detect dimming events')

    args = parser.parse_args()

    print(f"Loading FITS files from {args.input_dir}")
    frames, filenames, wcs_objects = load_fits_stack(args.input_dir)

    print(f"Detecting transients (threshold={args.threshold})")
    detections = detect_transients(
        frames, filenames, wcs_objects,
        z_thresh=args.threshold,
        detect_dimming=args.dimming,
        output_dir=args.output
    )

    print(f"Cross-matching with GAIA")
    matched_detections = crossmatch_with_gaia(detections)

    print(f"Found {len(matched_detections)} detections")
    for i, det in enumerate(matched_detections):
        status = "matched" if det.get('match_name') else "unmatched"
        confidence = det['confidence'] * 100
        print(f"  {i+1}: Frame {det['frame']}, confidence {confidence:.1f}% ({status})")

if __name__ == '__main__':
    main()
```

### Web API Integration

```python
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """REST API endpoint for detection"""
    try:
        # Get parameters
        threshold = float(request.form.get('threshold', 6.0))
        detect_dimming = request.form.get('dimming', 'false').lower() == 'true'

        # Handle file upload
        if 'fits_files' not in request.files:
            return jsonify({'error': 'No FITS files provided'}), 400

        files = request.files.getlist('fits_files')

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for file in files:
                if file.filename.endswith('.fits'):
                    file.save(os.path.join(temp_dir, file.filename))

            # Process
            frames, filenames, wcs_objects = load_fits_stack(temp_dir)
            detections = detect_transients(
                frames, filenames, wcs_objects,
                z_thresh=threshold,
                detect_dimming=detect_dimming,
                output_dir=temp_dir
            )

            # Return results
            return jsonify({
                'success': True,
                'detections': len(detections),
                'results': detections
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Testing

### Unit Tests

```python
import unittest
import numpy as np
from pulsehunter_core import detect_transients

class TestDetection(unittest.TestCase):

    def setUp(self):
        # Create synthetic test data
        self.frames = np.random.normal(1000, 50, (10, 100, 100)).astype(np.float32)
        self.filenames = [f'test_{i:03d}.fits' for i in range(10)]
        self.wcs_objects = [None] * 10

        # Add synthetic transient
        self.frames[5, 50, 50] = 5000

    def test_basic_detection(self):
        detections = detect_transients(
            self.frames, self.filenames, self.wcs_objects,
            z_thresh=3.0
        )
        self.assertGreater(len(detections), 0)

    def test_threshold_sensitivity(self):
        low_thresh = detect_transients(
            self.frames, self.filenames, self.wcs_objects, z_thresh=2.0
        )
        high_thresh = detect_transients(
            self.frames, self.filenames, self.wcs_objects, z_thresh=8.0
        )
        self.assertGreaterEqual(len(low_thresh), len(high_thresh))

if __name__ == '__main__':
    unittest.main()
```

For more examples and detailed documentation, visit the [PulseHunter GitHub repository](https://github.com/Kelsidavis/PulseHunter).
