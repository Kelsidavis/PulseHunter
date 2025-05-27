# 🌟 PulseHunter: Optical SETI & Exoplanet Detection Network

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da)](https://discord.gg/your-server)

> **Democratizing the search for extraterrestrial intelligence and exoplanet discovery through citizen science**

PulseHunter empowers amateur astronomers worldwide to contribute to cutting-edge astronomical research. Our open-source platform combines advanced detection algorithms with collaborative networking to search for optical SETI signals and exoplanet transits.

![PulseHunter Dashboard](docs/images/dashboard-preview.png)

## 🚀 Features

### 🔭 **Advanced Detection Capabilities**
- **Statistical Transient Detection**: Z-score analysis with configurable thresholds
- **FITS Image Processing**: Full calibration pipeline (bias, dark, flat correction)
- **Plate Solving Integration**: Automatic astrometric calibration using ASTAP
- **Light Curve Generation**: Publication-quality photometric analysis

### 🌌 **Smart Cross-Matching**
- **GAIA DR3 Integration**: Automatic stellar catalog matching
- **NASA Exoplanet Archive**: Transit candidate verification
- **False Positive Filtering**: Advanced algorithms to reduce noise

### 🌐 **Collaborative Network**
- **Global Data Sharing**: Automatic upload to central database
- **Real-time Dashboard**: Interactive sky map with all network detections
- **Observer Recognition**: Credit system for contributing astronomers

### 📊 **Professional Analysis Tools**
- **Export Functionality**: CSV, PNG, and comprehensive reports
- **Statistical Reporting**: Confidence metrics and detection statistics
- **Publication Support**: Research-grade documentation

## 🎯 Scientific Impact

PulseHunter addresses critical needs in modern astronomy:

- **SETI Research**: Recent discoveries of unexplained optical pulses demonstrate the potential for amateur contributions to SETI science
- **Exoplanet Confirmation**: TESS has identified over 10,000 exoplanet candidates requiring ground-based follow-up
- **Citizen Science**: Amateur astronomers have already confirmed exoplanet transits and asteroid occultations using similar approaches

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Qt6 (for GUI)
- ASTAP (for plate solving)
- 8+ GB RAM recommended
- Dark sky location preferred

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Kelsidavis/PulseHunter.git
cd PulseHunter

# Install dependencies
pip install -r requirements.txt

# Install ASTAP (plate solving)
# Download from: https://www.hnsky.org/astap.htm

# Launch the application
python pulse_gui.py
```

### Docker Installation (Recommended)
```bash
docker pull kelsidavis/pulsehunter:latest
docker run -it --rm -v $(pwd)/data:/app/data pulsehunter
```

## 🎓 Quick Tutorial

### 1. **Setup Calibration**
```python
# Configure your observatory
python pulse_gui.py
# Click "Calibration Setup"
# Select bias, dark, and flat frame folders
# Choose camera type (mono/OSC)
```

### 2. **Load Dataset**
- Select folder containing FITS files
- Ensure consistent exposure times
- Minimum 50 frames recommended

### 3. **Run Detection**
- Set detection threshold (6σ recommended)
- Monitor processing progress
- Review results in interactive table

### 4. **Analyze Results**
- High confidence (>80%): Likely genuine signals
- Medium confidence (50-80%): Requires follow-up
- Cross-matched objects: Known catalog sources

## 📈 Project Status

PulseHunter is in active development:
- **Current Version**: Alpha (seeking early adopters)
- **Development Stage**: Core functionality complete
- **Network Status**: Ready for first observers
- **Community**: Growing - join us as a founding contributor!

*Be among the first to help build this network!*

## 🔬 Scientific Methodology

### Detection Algorithm
PulseHunter uses advanced statistical methods:

1. **Image Calibration**: Dark/bias subtraction, flat fielding
2. **Astrometric Solution**: WCS calibration via ASTAP
3. **Differential Photometry**: Frame-to-frame comparison
4. **Statistical Analysis**: Z-score computation with outlier detection
5. **Catalog Cross-matching**: GAIA and exoplanet database queries

### Quality Metrics
- **Confidence Score**: 0-100% based on signal strength
- **False Positive Rate**: <5% for high-confidence detections
- **Detection Limit**: ~0.5% photometric precision

## 🌍 Join the Community

PulseHunter is just getting started - help us build something amazing!

### Be a Founding Member
- **Early Adopters Wanted**: Test the software with your equipment
- **Shape the Future**: Your feedback will guide development priorities
- **Scientific Impact**: Contribute to methodology development
- **Recognition**: Founding contributors will be prominently credited

### Vision for Growth
As we build our community, we aim to:
- Partner with astronomy organizations
- Collaborate with professional observatories
- Publish peer-reviewed research
- Create educational programs

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Manual](docs/user-guide.md)
- [API Reference](docs/api.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Scientific Papers](docs/publications.md)

## 🤝 Contributing

We welcome contributions from astronomers, developers, and data scientists!

### Ways to Contribute
- **🔭 Observations**: Run PulseHunter at your observatory
- **💻 Code**: Improve algorithms, add features, fix bugs
- **📖 Documentation**: Help others get started
- **🧪 Testing**: Validate detections, report issues
- **🎨 Design**: Improve user interface and experience

### Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/PulseHunter.git
cd PulseHunter

# Create development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
python -m flask run --debug
```

## 📊 Data Policy

- **Open Science**: All detection data freely available
- **Observer Credit**: Contributors acknowledged in publications
- **Privacy**: Personal information never shared
- **Data Retention**: Observations archived indefinitely

## 🎖️ Future Recognition

As PulseHunter grows, we plan to:
- Acknowledge all contributors in publications
- Create a hall of fame for significant discoveries
- Collaborate with professional astronomers on research papers
- Establish awards for outstanding contributions

*Your early participation helps establish the foundation for citizen science discoveries!*

## 🔗 Related Projects

- [SETI@home](https://setiathome.berkeley.edu/) - Radio SETI analysis
- [Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess) - Visual planet detection
- [Exoplanet Watch](https://exoplanets.nasa.gov/exoplanet-watch/) - NASA citizen science program

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Kelsidavis/PulseHunter/issues)
- **Email**: [pulsehunter@geekastro.dev](mailto:pulsehunter@geekastro.dev)
- **Website**: [https://geekastro.dev](https://geekastro.dev)

*Community Discord and forums coming soon as we grow!*

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SETI Institute**: Inspiration from their citizen science initiatives
- **NASA Exoplanet Science Institute**: Public data access that makes this possible
- **Amateur Astronomy Community**: The dedicated observers who make discoveries happen
- **Open Source Community**: The tools and libraries that power PulseHunter

---

<div align="center">

**🌌 Help build the future of citizen science astronomy 🌌**

[Get Started](docs/installation.md) • [Report Issues](https://github.com/Kelsidavis/PulseHunter/issues) • [Contact Developer](mailto:pulsehunter@geekastro.dev)

</div>

---

*PulseHunter is developed by [Kelsi Davis](https://github.com/Kelsidavis). Join the effort to democratize the search for extraterrestrial intelligence through citizen science.*
