"""
PulseHunter Main GUI Application - Fixed and Functional Version
Optical SETI and Exoplanet Transit Detection Pipeline
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import our fixed modules
import pulsehunter_core as core
from calibration_manager import CalibrationManager, quick_calibrate_folder

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """Worker thread for processing images"""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_message = pyqtSignal(str)
    processing_complete = pyqtSignal(list)  # List of detections
    error_occurred = pyqtSignal(str)

    def __init__(self, folder_path: str, settings: Dict):
        super().__init__()
        self.folder_path = folder_path
        self.settings = settings
        self._is_running = True

    def run(self):
        """Run the processing"""
        try:
            self.log_message.emit("Starting image processing...")
            self.status_updated.emit("Loading FITS files...")

            # Load calibrated FITS files
            frames, filenames, wcs_objects = core.load_calibrated_fits(
                self.folder_path,
                plate_solve_missing=self.settings.get("plate_solve", False),
                astap_exe=self.settings.get("astap_exe", "astap"),
                progress_callback=lambda p: self.progress_updated.emit(int(p * 0.5)),
            )

            if len(frames) == 0:
                self.error_occurred.emit("No valid FITS files found!")
                return

            self.log_message.emit(f"Loaded {len(frames)} FITS files")
            self.status_updated.emit("Detecting transients...")

            # Detect transients
            detections = core.detect_transients(
                frames,
                filenames,
                wcs_objects,
                output_dir=str(Path(self.folder_path).parent / "detections"),
                z_thresh=self.settings.get("z_threshold", 6.0),
                min_pixels=self.settings.get("min_pixels", 4),
                detect_dimming=self.settings.get("detect_dimming", True),
                progress_callback=lambda p: self.progress_updated.emit(
                    50 + int(p * 0.4)
                ),
            )

            self.log_message.emit(f"Found {len(detections)} potential detections")

            # Save report
            self.status_updated.emit("Saving results...")
            report_path = Path(self.folder_path).parent / "detection_report.json"
            core.save_detection_report(detections, str(report_path))

            # Create detection images
            if detections and self.settings.get("create_images", True):
                self.status_updated.emit("Creating detection images...")
                core.create_detection_images(
                    detections,
                    frames,
                    str(Path(self.folder_path).parent / "detection_images"),
                )

            self.progress_updated.emit(100)
            self.status_updated.emit("Processing complete!")
            self.processing_complete.emit(detections)

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

    def stop(self):
        """Stop the processing"""
        self._is_running = False


class CalibrationWorker(QThread):
    """Worker thread for calibration"""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_message = pyqtSignal(str)
    calibration_complete = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, settings: Dict):
        super().__init__()
        self.settings = settings

    def run(self):
        """Run the calibration"""
        try:
            self.log_message.emit("Starting calibration process...")

            success = quick_calibrate_folder(
                lights_folder=self.settings["lights_folder"],
                bias_folder=self.settings.get("bias_folder"),
                dark_folder=self.settings.get("dark_folder"),
                flat_folder=self.settings.get("flat_folder"),
                output_folder=self.settings.get("output_folder"),
            )

            if success:
                self.progress_updated.emit(100)
                self.status_updated.emit("Calibration complete!")
                self.calibration_complete.emit(True)
            else:
                self.error_occurred.emit("Calibration failed - check logs")

        except Exception as e:
            logger.error(f"Calibration error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))


class DetectionViewer(QWidget):
    """Widget for viewing detection results"""

    def __init__(self):
        super().__init__()
        self.detections = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Brightening", "Dimming", "High Confidence"])
        self.filter_combo.currentTextChanged.connect(self.update_display)
        controls.addWidget(self.filter_combo)

        controls.addWidget(QLabel("Min Z-score:"))
        self.min_z_spin = QDoubleSpinBox()
        self.min_z_spin.setRange(0, 50)
        self.min_z_spin.setValue(6)
        self.min_z_spin.valueChanged.connect(self.update_display)
        controls.addWidget(self.min_z_spin)
        controls.addStretch()

        # --- PATCH: controls bar is now in a fixed-height widget
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setFixedHeight(40)  # or adjust as needed
        layout.addWidget(controls_widget)

        # Results table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Plot area
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def set_detections(self, detections: List[Dict]):
        """Set the detections to display"""
        self.detections = detections
        self.update_display()

    def update_display(self):
        """Update the display with filtered detections"""
        # Filter detections
        filtered = []
        filter_type = self.filter_combo.currentText()
        min_z = self.min_z_spin.value()

        for det in self.detections:
            if abs(det.get("z_score", 0)) < min_z:
                continue

            if filter_type == "Brightening" and det.get("dimming", False):
                continue
            elif filter_type == "Dimming" and not det.get("dimming", False):
                continue
            elif filter_type == "High Confidence" and det.get("confidence", 0) < 0.8:
                continue

            filtered.append(det)

        # Update table
        self.table.setRowCount(len(filtered))
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Frame", "File", "X", "Y", "Z-Score", "Type", "Confidence"]
        )

        for i, det in enumerate(filtered):
            self.table.setItem(i, 0, QTableWidgetItem(str(det.get("frame", ""))))
            self.table.setItem(
                i, 1, QTableWidgetItem(Path(det.get("filename", "")).name)
            )
            self.table.setItem(i, 2, QTableWidgetItem(str(det.get("x", ""))))
            self.table.setItem(i, 3, QTableWidgetItem(str(det.get("y", ""))))
            self.table.setItem(i, 4, QTableWidgetItem(f"{det.get('z_score', 0):.1f}"))
            self.table.setItem(
                i,
                5,
                QTableWidgetItem("Dimming" if det.get("dimming") else "Brightening"),
            )
            self.table.setItem(
                i, 6, QTableWidgetItem(f"{det.get('confidence', 0):.0%}")
            )

        self.table.resizeColumnsToContents()

        # Update plot
        self.plot_detection_summary(filtered)

    def plot_detection_summary(self, detections):
        """Plot summary of detections"""
        self.figure.clear()

        if not detections:
            return

        ax = self.figure.add_subplot(111)

        # Extract data
        z_scores = [abs(d.get("z_score", 0)) for d in detections]
        frames = [d.get("frame", 0) for d in detections]
        types = ["Dimming" if d.get("dimming") else "Brightening" for d in detections]

        # Color by type
        colors = ["blue" if t == "Dimming" else "red" for t in types]

        # Plot
        scatter = ax.scatter(frames, z_scores, c=colors, alpha=0.6, s=50)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("|Z-Score|")
        ax.set_title(f"Detection Summary ({len(detections)} detections)")
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label="Brightening"),
            Patch(facecolor="blue", label="Dimming"),
        ]
        ax.legend(handles=legend_elements)

        self.figure.tight_layout()
        self.canvas.draw()


class PulseHunterMainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.settings = QSettings("PulseHunter", "MainApp")
        self.calibration_manager = CalibrationManager()
        self.current_detections = []

        self.setup_ui()
        self.setup_menus()
        self.restore_settings()

        # --- PulseHunter Web Theme QSS ---
        self.setStyleSheet(
            """
            QWidget {
                background-color: #151c2b;
                color: #e0e6f0;
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 15px;
            }
            QMainWindow, QDialog, QTabWidget::pane {
                background: #181e2f;
                border-radius: 16px;
            }
            QTabWidget::pane {
                border-top: 2px solid #297fff;
                top: -1.5em;
                padding-top: 32px;  /* Increased vertical space between tab bar and tab page */
                margin-top: 0px;
            }
            QGroupBox {
                background: #202942;
                border: 1px solid #297fff;
                border-radius: 12px;
                margin-top: 18px;
                padding: 10px;
            }
            QGroupBox:title {
                color: #68a0ff;
                padding-left: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QLabel {
                color: #e6ebff;
            }
            QPushButton {
                background-color: #297fff;
                color: #f7faff;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: 600;
                font-size: 15px;
                transition: background 0.2s;
            }
            QPushButton:hover {
                background-color: #379fff;
            }
            QPushButton:pressed {
                background-color: #235db2;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #232d43;
                color: #e0e6f0;
                border: 1px solid #297fff;
                border-radius: 7px;
                padding: 4px 8px;
            }
            QProgressBar {
                background: #232d43;
                color: #68a0ff;
                border: 1px solid #297fff;
                border-radius: 7px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #297fff;
                border-radius: 7px;
            }
            QTableWidget, QHeaderView::section {
                background: #181e2f;
                color: #e0e6f0;
                border: 1px solid #297fff;
            }
            QTabWidget::pane {
                border-top: 2px solid #297fff;
                top: -1.5em;
            }
            QTabBar::tab {
                background: #202942;
                color: #b68ff9;
                border-radius: 9px 9px 0 0;
                padding: 7px 25px;
                margin-right: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: #232d43;
                color: #f7faff;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #232d43;
                border: none;
                width: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #379fff;
                min-height: 24px;
                border-radius: 6px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
                border: none;
            }
            QFrame {
                background: #202942;
                border-radius: 12px;
                border: 1px solid #232d43;
            }
        """
        )
        # Optionally set global font for more consistency
        from PyQt6.QtGui import QFont

        font = QFont("Segoe UI", 10)
        from PyQt6.QtWidgets import QApplication

        QApplication.instance().setFont(font)

    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("PulseHunter - Optical SETI & Exoplanet Detection")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Welcome tab
        self.setup_welcome_tab()

        # Calibration tab
        self.setup_calibration_tab()

        # Processing tab
        self.setup_processing_tab()

        # Results tab
        self.setup_results_tab()

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def setup_welcome_tab(self):
        """Setup welcome tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Welcome to PulseHunter")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setPointSize(24)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "PulseHunter is an astronomical image analysis tool for:\n\n"
            "• Optical SETI - Search for artificial signals\n"
            "• Exoplanet detection - Find transiting exoplanets\n"
            "• Transient detection - Discover variable stars and other phenomena\n\n"
            "Get started by calibrating your images, then process them to find anomalies!"
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Quick start buttons
        button_layout = QHBoxLayout()

        calibrate_btn = QPushButton("Start Calibration")
        calibrate_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        button_layout.addWidget(calibrate_btn)

        process_btn = QPushButton("Process Images")
        process_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        button_layout.addWidget(process_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        self.tabs.addTab(widget, "Welcome")

    def setup_calibration_tab(self):
        """Setup calibration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Folder selection
        folders_group = QGroupBox("Select Calibration Folders")
        folders_layout = QGridLayout(folders_group)

        # Lights folder
        folders_layout.addWidget(QLabel("Lights Folder:"), 0, 0)
        self.lights_folder_edit = QLineEdit()
        folders_layout.addWidget(self.lights_folder_edit, 0, 1)
        browse_lights_btn = QPushButton("Browse...")
        browse_lights_btn.clicked.connect(lambda: self.browse_folder("lights"))
        folders_layout.addWidget(browse_lights_btn, 0, 2)

        # Bias folder
        folders_layout.addWidget(QLabel("Bias Folder:"), 1, 0)
        self.bias_folder_edit = QLineEdit()
        folders_layout.addWidget(self.bias_folder_edit, 1, 1)
        browse_bias_btn = QPushButton("Browse...")
        browse_bias_btn.clicked.connect(lambda: self.browse_folder("bias"))
        folders_layout.addWidget(browse_bias_btn, 1, 2)

        # Dark folder
        folders_layout.addWidget(QLabel("Dark Folder:"), 2, 0)
        self.dark_folder_edit = QLineEdit()
        folders_layout.addWidget(self.dark_folder_edit, 2, 1)
        browse_dark_btn = QPushButton("Browse...")
        browse_dark_btn.clicked.connect(lambda: self.browse_folder("dark"))
        folders_layout.addWidget(browse_dark_btn, 2, 2)

        # Flat folder
        folders_layout.addWidget(QLabel("Flat Folder:"), 3, 0)
        self.flat_folder_edit = QLineEdit()
        folders_layout.addWidget(self.flat_folder_edit, 3, 1)
        browse_flat_btn = QPushButton("Browse...")
        browse_flat_btn.clicked.connect(lambda: self.browse_folder("flat"))
        folders_layout.addWidget(browse_flat_btn, 3, 2)

        layout.addWidget(folders_group)

        # Calibration controls
        controls_layout = QHBoxLayout()

        self.calibrate_btn = QPushButton("Start Calibration")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        controls_layout.addWidget(self.calibrate_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Progress
        self.cal_progress = QProgressBar()
        self.cal_progress.setVisible(False)
        layout.addWidget(self.cal_progress)

        # Log
        self.cal_log = QTextEdit()
        self.cal_log.setReadOnly(True)
        self.cal_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.cal_log)

        self.tabs.addTab(widget, "Calibration")

    def setup_processing_tab(self):
        """Setup processing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Folder selection
        folder_group = QGroupBox("Select Folder to Process")
        folder_layout = QHBoxLayout(folder_group)

        folder_layout.addWidget(QLabel("Calibrated Images:"))
        self.process_folder_edit = QLineEdit()
        folder_layout.addWidget(self.process_folder_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_process_folder)
        folder_layout.addWidget(browse_btn)

        layout.addWidget(folder_group)

        # Processing settings
        settings_group = QGroupBox("Processing Settings")
        settings_layout = QGridLayout(settings_group)

        settings_layout.addWidget(QLabel("Z-Score Threshold:"), 0, 0)
        self.z_threshold_spin = QDoubleSpinBox()
        self.z_threshold_spin.setRange(3.0, 20.0)
        self.z_threshold_spin.setValue(6.0)
        self.z_threshold_spin.setSingleStep(0.5)
        settings_layout.addWidget(self.z_threshold_spin, 0, 1)

        settings_layout.addWidget(QLabel("Min Pixels:"), 1, 0)
        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setRange(1, 100)
        self.min_pixels_spin.setValue(4)
        settings_layout.addWidget(self.min_pixels_spin, 1, 1)

        self.detect_dimming_check = QCheckBox("Detect Dimming Events")
        self.detect_dimming_check.setChecked(True)
        settings_layout.addWidget(self.detect_dimming_check, 2, 0, 1, 2)

        self.create_images_check = QCheckBox("Create Detection Images")
        self.create_images_check.setChecked(True)
        settings_layout.addWidget(self.create_images_check, 3, 0, 1, 2)

        self.plate_solve_check = QCheckBox("Plate Solve (requires ASTAP)")
        settings_layout.addWidget(self.plate_solve_check, 4, 0, 1, 2)

        layout.addWidget(settings_group)

        # Process button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Progress
        self.process_progress = QProgressBar()
        self.process_progress.setVisible(False)
        layout.addWidget(self.process_progress)

        # Log
        self.process_log = QTextEdit()
        self.process_log.setReadOnly(True)
        self.process_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.process_log)

        self.tabs.addTab(widget, "Processing")

    def setup_results_tab(self):
        """Setup results tab"""
        self.detection_viewer = DetectionViewer()
        self.tabs.addTab(self.detection_viewer, "Results")

    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_results_action = QAction("Load Results...", self)
        load_results_action.triggered.connect(self.load_results)
        file_menu.addAction(load_results_action)

        export_results_action = QAction("Export Results...", self)
        export_results_action.triggered.connect(self.export_results)
        file_menu.addAction(export_results_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def browse_folder(self, folder_type: str):
        """Browse for a folder"""
        folder = QFileDialog.getExistingDirectory(
            self, f"Select {folder_type.title()} Folder"
        )
        if folder:
            if folder_type == "lights":
                self.lights_folder_edit.setText(folder)
            elif folder_type == "bias":
                self.bias_folder_edit.setText(folder)
            elif folder_type == "dark":
                self.dark_folder_edit.setText(folder)
            elif folder_type == "flat":
                self.flat_folder_edit.setText(folder)

    def browse_process_folder(self):
        """Browse for folder to process"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Calibrated Images Folder"
        )
        if folder:
            self.process_folder_edit.setText(folder)

    def start_calibration(self):
        """Start the calibration process"""
        lights_folder = self.lights_folder_edit.text()
        if not lights_folder:
            QMessageBox.warning(
                self, "No Lights Folder", "Please select a lights folder!"
            )
            return

        # Prepare settings
        settings = {
            "lights_folder": lights_folder,
            "bias_folder": self.bias_folder_edit.text() or None,
            "dark_folder": self.dark_folder_edit.text() or None,
            "flat_folder": self.flat_folder_edit.text() or None,
        }

        # Start worker
        self.cal_worker = CalibrationWorker(settings)
        self.cal_worker.progress_updated.connect(self.cal_progress.setValue)
        self.cal_worker.status_updated.connect(self.status_bar.showMessage)
        self.cal_worker.log_message.connect(self.cal_log.append)
        self.cal_worker.calibration_complete.connect(self.calibration_complete)
        self.cal_worker.error_occurred.connect(self.calibration_error)

        # Update UI
        self.calibrate_btn.setEnabled(False)
        self.cal_progress.setVisible(True)
        self.cal_progress.setValue(0)

        self.cal_worker.start()

    def calibration_complete(self, success: bool):
        """Handle calibration completion"""
        self.calibrate_btn.setEnabled(True)
        self.cal_progress.setVisible(False)

        if success:
            QMessageBox.information(
                self,
                "Calibration Complete",
                "Calibration completed successfully!\n\n"
                "Calibrated images have been saved to the 'calibrated' folder.",
            )

            # Auto-fill process folder
            lights_folder = Path(self.lights_folder_edit.text())
            calibrated_folder = lights_folder.parent / "calibrated"
            if calibrated_folder.exists():
                self.process_folder_edit.setText(str(calibrated_folder))
                self.tabs.setCurrentIndex(2)  # Switch to processing tab

    def calibration_error(self, error: str):
        """Handle calibration error"""
        self.calibrate_btn.setEnabled(True)
        self.cal_progress.setVisible(False)
        QMessageBox.critical(self, "Calibration Error", f"Calibration failed:\n{error}")

    def start_processing(self):
        """Start processing images"""
        folder = self.process_folder_edit.text()
        if not folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder to process!")
            return

        # Prepare settings
        settings = {
            "z_threshold": self.z_threshold_spin.value(),
            "min_pixels": self.min_pixels_spin.value(),
            "detect_dimming": self.detect_dimming_check.isChecked(),
            "create_images": self.create_images_check.isChecked(),
            "plate_solve": self.plate_solve_check.isChecked(),
            "astap_exe": self.settings.value("astap_exe", "astap"),
        }

        # Start worker
        self.process_worker = ProcessingWorker(folder, settings)
        self.process_worker.progress_updated.connect(self.process_progress.setValue)
        self.process_worker.status_updated.connect(self.status_bar.showMessage)
        self.process_worker.log_message.connect(self.process_log.append)
        self.process_worker.processing_complete.connect(self.processing_complete)
        self.process_worker.error_occurred.connect(self.processing_error)

        # Update UI
        self.process_btn.setEnabled(False)
        self.process_progress.setVisible(True)
        self.process_progress.setValue(0)

        self.process_worker.start()

    def processing_complete(self, detections: List[Dict]):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.process_progress.setVisible(False)

        self.current_detections = detections
        self.detection_viewer.set_detections(detections)

        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processing completed!\n\n"
            f"Found {len(detections)} detections.\n\n"
            f"Results saved to detection_report.json",
        )

        # Switch to results tab
        self.tabs.setCurrentIndex(3)

    def processing_error(self, error: str):
        """Handle processing error"""
        self.process_btn.setEnabled(True)
        self.process_progress.setVisible(False)
        QMessageBox.critical(self, "Processing Error", f"Processing failed:\n{error}")

    def load_results(self):
        """Load results from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Results", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                detections = data.get("detections", [])
                self.current_detections = detections
                self.detection_viewer.set_detections(detections)
                self.tabs.setCurrentIndex(3)

            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error", f"Failed to load results:\n{e}"
                )

    def export_results(self):
        """Export results to file"""
        if not self.current_detections:
            QMessageBox.information(self, "No Results", "No results to export!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "detections.csv", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                import csv

                with open(file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["Frame", "File", "X", "Y", "Z-Score", "Type", "Confidence"]
                    )

                    for det in self.current_detections:
                        writer.writerow(
                            [
                                det.get("frame", ""),
                                Path(det.get("filename", "")).name,
                                det.get("x", ""),
                                det.get("y", ""),
                                det.get("z_score", 0),
                                "Dimming" if det.get("dimming") else "Brightening",
                                det.get("confidence", 0),
                            ]
                        )

                QMessageBox.information(
                    self, "Export Complete", "Results exported successfully!"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export results:\n{e}"
                )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About PulseHunter",
            (
                "<div style='background:#151c2b;padding:12px 8px 12px 8px;'>"
                "<h2 style='color:#297fff; margin: 0 0 8px 0; font-size: 1.6em;'>PulseHunter</h2>"
                "<div style='color:#b68ff9;font-weight:bold; margin-bottom: 8px;'>"
                "Optical SETI &amp; Exoplanet Detection Pipeline"
                "</div>"
                "<div style='color:#e0e6f0; margin-bottom:10px;'>"
                "Automated search for:<br>"
                "&nbsp;&nbsp;&bull; Artificial signals (Optical SETI)<br>"
                "&nbsp;&nbsp;&bull; Transiting exoplanets<br>"
                "&nbsp;&nbsp;&bull; Variable stars and transients"
                "</div>"
                "<div style='margin:14px 0 6px 0; font-size:1.08em;'>"
                "<span style='color:#e0e6f0;'>Created by</span> "
                "<span style='color:#68a0ff;font-weight:bold;'>Kelsi Davis</span>"
                "</div>"
                "<div style='margin-bottom:8px; font-size:0.98em;'>"
                "<a href='https://geekastro.dev/pulsehunter/' style='color:#379fff; text-decoration:none;'>"
                "🌐 Project Website</a><br>"
                "<a href='https://github.com/Kelsidavis/PulseHunter' style='color:#b68ff9; text-decoration:none;'>"
                "🐙 GitHub Repository</a>"
                "</div>"
                "<div style='margin-top:14px; color:#888; font-size:0.97em;'>"
                "&copy; 2024 Kelsi Davis &mdash; All Rights Reserved"
                "</div>"
                "</div>"
            ),
        )

    def restore_settings(self):
        """Restore saved settings"""
        # Window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Folder paths
        self.lights_folder_edit.setText(self.settings.value("lights_folder", ""))
        self.bias_folder_edit.setText(self.settings.value("bias_folder", ""))
        self.dark_folder_edit.setText(self.settings.value("dark_folder", ""))
        self.flat_folder_edit.setText(self.settings.value("flat_folder", ""))
        self.process_folder_edit.setText(self.settings.value("process_folder", ""))

        # Processing settings
        self.z_threshold_spin.setValue(float(self.settings.value("z_threshold", 6.0)))
        self.min_pixels_spin.setValue(int(self.settings.value("min_pixels", 4)))
        self.detect_dimming_check.setChecked(
            self.settings.value("detect_dimming", True, type=bool)
        )
        self.create_images_check.setChecked(
            self.settings.value("create_images", True, type=bool)
        )
        self.plate_solve_check.setChecked(
            self.settings.value("plate_solve", False, type=bool)
        )

    def closeEvent(self, event):
        """Save settings on close"""
        # Save geometry
        self.settings.setValue("geometry", self.saveGeometry())

        # Save folder paths
        self.settings.setValue("lights_folder", self.lights_folder_edit.text())
        self.settings.setValue("bias_folder", self.bias_folder_edit.text())
        self.settings.setValue("dark_folder", self.dark_folder_edit.text())
        self.settings.setValue("flat_folder", self.flat_folder_edit.text())
        self.settings.setValue("process_folder", self.process_folder_edit.text())

        # Save processing settings
        self.settings.setValue("z_threshold", self.z_threshold_spin.value())
        self.settings.setValue("min_pixels", self.min_pixels_spin.value())
        self.settings.setValue("detect_dimming", self.detect_dimming_check.isChecked())
        self.settings.setValue("create_images", self.create_images_check.isChecked())
        self.settings.setValue("plate_solve", self.plate_solve_check.isChecked())

        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("PulseHunter")
    app.setOrganizationName("AstronomyTools")

    # Set dark theme
    app.setStyle("Fusion")

    window = PulseHunterMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
