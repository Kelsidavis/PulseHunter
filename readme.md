# 🔭 PulseHunter

**PulseHunter** is an open-source Optical SETI and exoplanet transit detection pipeline for amateur astronomers and community scientists. It analyzes astronomical FITS data to identify candidate optical transients, dimming events, and potential exoplanetary transits using robust statistical and astrometric methods.

---

## ✨ Features

- 📁 Batch processing of FITS image stacks
- 🧠 Automatic detection of transient and dimming events (z-score based)
- 🌌 WCS-based astrometry with optional plate solving (ASTAP)
- 🌟 GAIA DR3 cross-matching for object classification and distance
- 🪐 Exoplanet transit correlation using NASA Exoplanet Archive
- 📈 Light curve extraction and visualization
- 🗺️ Aladin Lite sky viewer with interactive report map
- 📊 Web dashboard and JSON-based detection reporting

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip packages: `astropy`, `numpy`, `opencv-python`, `astroquery`, `matplotlib`, `requests`, `PySide6`
- Optional: [ASTAP](https://www.hnsky.org/astap.htm) for plate solving

### Setup

```bash
git clone https://github.com/Kelsidavis/PulseHunter.git
cd PulseHunter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Usage

1. Launch the GUI:

```bash
python pulse_gui.py
```

2. Select a folder of calibrated FITS light frames.
3. Set detection threshold and run analysis.
4. Review detections, preview cutouts, export reports.
5. Web dashboard (`index.html`) shows uploaded events; `exoplanet_transits.html` shows potential planetary transits.

---

## 📂 Folder Structure

```
pulsehunter/
├── pulse_gui.py               # Main GUI
├── pulsehunter_core.py        # Calibration + detection logic
├── calibration.py             # Master frame builder + report packager
├── exoplanet_match.py         # Transit correlation via NASA Archive
├── reports/                   # Uploaded detection reports (JSON)
├── index.html                 # Web dashboard
├── exoplanet_transits.html    # Exoplanet-specific event viewer
├── style.css                  # Dashboard styling
├── submit_report.php          # Server-side report receiver
```

---

## 🌐 Live Demo

> 📡 Visit the live dashboard:  
[https://geekastro.dev/pulsehunter](https://geekastro.dev/pulsehunter)

---

## 🙌 Credits

Created and maintained by **Kelsi Davis**  
🔗 [geekastro.dev](https://geekastro.dev) | 📧 dumbandroid@gmail.com

---

## 🛡 License

MIT License – free for academic, educational, and community use.
