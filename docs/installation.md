# PulseHunter Installation Guide

This guide will help you install and configure PulseHunter on your system.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB free space for software and data
- **Internet**: Required for catalog queries and data uploads

### Recommended Hardware
- **CPU**: Multi-core processor (quad-core or better)
- **RAM**: 32GB+ for large datasets
- **Storage**: SSD with 100GB+ free space
- **Camera**: CCD/CMOS camera with FITS output
- **Telescope**: 6" aperture minimum for best results

## Quick Installation

### Option 1: pip install (Recommended)
```bash
pip install pulsehunter
```

### Option 2: From Source
```bash
git clone https://github.com/Kelsidavis/PulseHunter.git
cd PulseHunter
pip install -r requirements.txt
python pulse_gui.py
```

### Option 3: Docker
```bash
docker pull kelsidavis/pulsehunter:latest
docker run -it --rm -v $(pwd)/data:/app/data pulsehunter
```

## Detailed Installation Steps

### 1. Install Python Dependencies

First, ensure you have Python 3.8+ installed:
```bash
python --version  # Should show 3.8 or higher
```

Install required packages:
```bash
pip install numpy astropy matplotlib PySide6 opencv-python requests astroquery scipy pillow pytz
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### 2. Install ASTAP (Plate Solving)

PulseHunter uses ASTAP for astrometric calibration.

#### Windows
1. Download ASTAP from: https://www.hnsky.org/astap.htm
2. Install to default location (C:\Program Files\astap)
3. Add to PATH or note installation directory

#### macOS
```bash
# Using Homebrew
brew install astap

# Or download from website and install manually
```

#### Linux (Ubuntu/Debian)
```bash
# Download and install
wget https://www.hnsky.org/astap_cli.deb
sudo dpkg -i astap_cli.deb

# Or compile from source
git clone https://github.com/han-k59/astap.git
cd astap
make astap_cli
sudo cp astap_cli /usr/local/bin/astap
```

### 3. Download Star Catalogs

ASTAP requires star catalogs for plate solving:

```bash
# Download basic catalogs (required)
astap -download_tycho2
astap -download_ucac4

# Download additional catalogs (optional, for better precision)
astap -download_gaia
astap -download_2mass
```

### 4. Verify Installation

Test your installation:
```bash
python -c "import pulsehunter_core; print('Core module: OK')"
python -c "import calibration; print('Calibration module: OK')"
python -c "from PySide6.QtWidgets import QApplication; print('Qt GUI: OK')"
astap -version  # Should show ASTAP version
```

## Configuration

### First-Time Setup

1. **Launch PulseHunter**:
```bash
python pulse_gui.py
```

2. **Configure ASTAP Path**:
   - Click "Calibration Setup"
   - Click "Select ASTAP Executable"
   - Browse to your ASTAP installation

3. **Set Observer Information**:
   - Enter your name/observatory
   - This will be included in reports

### Observatory Configuration

#### Camera Settings
- **Monochrome**: Select for B&W CCD/CMOS cameras
- **One Shot Color**: Select for color cameras (Bayer matrix)

#### Calibration Frames
Create master calibration frames for best results:

**Bias Frames**:
- 20-50 frames minimum
- Shortest exposure (usually 0.001s)
- Same temperature as science frames
- Camera shutter closed

**Dark Frames**:
- 10-20 frames minimum
- Same exposure time as science frames
- Same temperature as science frames
- Camera shutter closed

**Flat Frames**:
- 10-20 frames minimum
- Twilight sky or light panel
- Same filter as science frames
- Uniform illumination

### Data Organization

Organize your files as follows:
```
Observatory/
├── calibration/
│   ├── bias/
│   │   ├── bias_001.fits
│   │   └── bias_002.fits
│   ├── dark/
│   │   ├── dark_001.fits
│   │   └── dark_002.fits
│   └── flat/
│       ├── flat_001.fits
│       └── flat_002.fits
└── science/
    ├── 2023-01-01/
    │   ├── target_001.fits
    │   ├── target_002.fits
    │   └── target_003.fits
    └── 2023-01-02/
        ├── target_001.fits
        └── target_002.fits
```

## Troubleshooting

### Common Issues

#### "ASTAP not found"
**Solution**:
- Verify ASTAP installation
- Check PATH environment variable
- Manually specify ASTAP path in configuration

#### "Qt platform plugin could not be initialized"
**Solution (Linux)**:
```bash
sudo apt-get install libgl1-mesa-glx libxkbcommon-x11-0
export QT_QPA_PLATFORM=xcb
```

#### "Permission denied" errors
**Solution**:
- Run with appropriate permissions
- Check file/folder ownership
- Avoid system directories for data

#### Memory errors with large datasets
**Solution**:
- Process smaller batches of images
- Increase system virtual memory
- Use 64-bit Python installation
- Consider cloud processing for very large datasets

#### Slow processing
**Solution**:
- Enable GPU acceleration if available
- Use SSD storage for temporary files
- Process during low system usage
- Consider distributed processing

### Performance Optimization

#### For Large Datasets
```bash
# Process in batches
python pulse_gui.py --batch-size 50

# Use multiple cores
python pulse_gui.py --threads 4

# Reduce memory usage
python pulse_gui.py --low-memory
```

#### For Slow Systems
- Reduce detection threshold to limit candidates
- Use smaller cutout sizes
- Skip expensive crossmatching for initial runs
- Process overnight for large surveys

### Getting Help

If you encounter issues:

1. **Check the FAQ**: [docs/faq.md](faq.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/Kelsidavis/PulseHunter/issues)
3. **Create new issue**: Include system info, error messages, and steps to reproduce
4. **Email support**: [pulsehunter@geekastro.dev](mailto:pulsehunter@geekastro.dev)

### System Information for Bug Reports

When reporting issues, include this information:
```bash
# System info
python --version
pip list | grep -E "(numpy|astropy|PySide6|opencv)"
astap -version

# On Linux/Mac
uname -a
cat /proc/meminfo | head -3  # Linux
system_profiler SPMemoryDataType  # Mac

# On Windows
systeminfo | findstr /C:"Total Physical Memory"
```

## Advanced Installation

### Development Installation

For contributors and developers:
```bash
git clone https://github.com/Kelsidavis/PulseHunter.git
cd PulseHunter
pip install -r requirements-dev.txt
pip install -e .  # Editable installation

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pulsehunter --cov-report=html
```

### Server Installation

For headless operation on servers:
```bash
# Install without Qt (command-line only)
pip install pulsehunter[headless]

# Run in batch mode
python -m pulsehunter.batch_process /path/to/fits/files

# Set up as service (Linux)
sudo cp scripts/pulsehunter.service /etc/systemd/system/
sudo systemctl enable pulsehunter
sudo systemctl start pulsehunter
```

### Cloud Installation

#### AWS EC2
```bash
# Launch instance with sufficient RAM
# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip -y
pip3 install pulsehunter

# Configure storage
sudo mkfs -t ext4 /dev/xvdf
sudo mount /dev/xvdf /data
```

#### Google Cloud Platform
```bash
# Create compute instance
gcloud compute instances create pulsehunter-vm \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB

# SSH and install
gcloud compute ssh pulsehunter-vm
curl -sSL https://install.python-poetry.org | python3 -
poetry install pulsehunter
```

## Next Steps

After successful installation:

1. **Read the User Guide**: [docs/user-guide.md](user-guide.md)
2. **Try the Tutorial**: Process sample data
3. **Join the Community**: Connect with other observers
4. **Start Observing**: Begin your first detection runs!

## Updates

Keep PulseHunter updated:
```bash
# Update via pip
pip install --upgrade pulsehunter

# Update from git
git pull origin main
pip install -r requirements.txt
```

Subscribe to releases on GitHub to get notifications of new versions.
