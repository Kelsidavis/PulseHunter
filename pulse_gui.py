"""
PulseHunter Main GUI Application - Enhanced and Fully Functional
Optical SETI and Exoplanet Transit Detection Pipeline GUI
"""

import sys
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMenuBar, QVBoxLayout, QWidget, QStatusBar, 
    QMessageBox, QLabel, QHBoxLayout, QGroupBox, QPushButton, QTextEdit, 
    QSplitter, QTabWidget, QProgressBar, QFileDialog, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QFrame, QGridLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QScrollArea
)
from PyQt6.QtGui import QAction, QIcon, QFont, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import (
    QSettings, QTimer, Qt, QThread, pyqtSignal, QObject, QMutex, QSize
)

# Import matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Import PulseHunter components
from enhanced_calibration_dialog import CalibrationSetupDialog
from calibration_utilities import (
    CalibrationConfig, ASTAPManager, CalibrationLogger, DialogPositionManager
)
import pulsehunter_core
from fits_processing import FITSProcessor, CalibrationProcessor
from exoplanet_match import match_transits_with_exoplanets

class ProcessingWorker(QThread):
    """Worker thread for image processing to prevent GUI freezing"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    detection_found = pyqtSignal(dict)
    processing_finished = pyqtSignal(bool, str, list)  # success, message, detections
    
    def __init__(self, project_data):
        super().__init__()
        self.project_data = project_data
        self.is_cancelled = False
        self.mutex = QMutex()
        
    def run(self):
        """Main processing function"""
        try:
            self.status_updated.emit("Initializing processing...")
            self.log_updated.emit("Starting PulseHunter processing pipeline...")
            
            # Extract project parameters
            input_folder = self.project_data.get('input_folder')
            detection_threshold = self.project_data.get('detection_threshold', 6.0)
            astap_path = self.project_data.get('astap_path', 'astap')
            
            # Calibration frames
            master_bias = self.project_data.get('master_bias')
            master_dark = self.project_data.get('master_dark') 
            master_flat = self.project_data.get('master_flat')
            
            if not input_folder or not Path(input_folder).exists():
                self.processing_finished.emit(False, "Invalid input folder", [])
                return
                
            self.progress_updated.emit(10)
            self.status_updated.emit("Loading FITS files...")
            self.log_updated.emit(f"Loading FITS files from: {input_folder}")
            
            # Load FITS stack using pulsehunter_core
            frames, filenames, wcs_objects = pulsehunter_core.load_fits_stack(
                input_folder,
                plate_solve_missing=True,
                astap_exe=astap_path,
                master_bias=master_bias,
                master_dark=master_dark,
                master_flat=master_flat
            )
            
            if len(frames) == 0:
                self.processing_finished.emit(False, "No valid FITS files loaded", [])
                return
                
            self.progress_updated.emit(30)
            self.log_updated.emit(f"Loaded {len(frames)} FITS files successfully")
            
            if self.is_cancelled:
                return
                
            self.status_updated.emit("Detecting transients...")
            self.log_updated.emit(f"Analyzing images with threshold z={detection_threshold}")
            
            # Detect transients
            detections = pulsehunter_core.detect_transients(
                frames, filenames, wcs_objects,
                output_dir=str(Path(input_folder).parent / "detections"),
                z_thresh=detection_threshold,
                cutout_size=50,
                edge_margin=20,
                detect_dimming=True
            )
            
            self.progress_updated.emit(60)
            self.log_updated.emit(f"Found {len(detections)} potential detections")
            
            if self.is_cancelled:
                return
                
            # Cross-match with GAIA if detections found
            if detections:
                self.status_updated.emit("Cross-matching with GAIA DR3...")
                self.log_updated.emit("Querying GAIA DR3 catalog for source matches...")
                
                try:
                    matched_detections = pulsehunter_core.crossmatch_with_gaia(
                        detections, radius_arcsec=5.0
                    )
                    detections = matched_detections
                    self.progress_updated.emit(80)
                except Exception as e:
                    self.log_updated.emit(f"GAIA cross-match failed: {e}")
                    
            if self.is_cancelled:
                return
                
            # Cross-match with exoplanet catalog
            if detections:
                self.status_updated.emit("Cross-matching with exoplanet catalog...")
                self.log_updated.emit("Checking NASA Exoplanet Archive...")
                
                try:
                    exo_matched = match_transits_with_exoplanets(
                        detections, radius_arcsec=10.0
                    )
                    detections = exo_matched
                    self.progress_updated.emit(90)
                except Exception as e:
                    self.log_updated.emit(f"Exoplanet cross-match failed: {e}")
                    
            # Generate summary statistics
            stats = pulsehunter_core.generate_summary_stats(detections)
            self.log_updated.emit(f"Processing complete: {stats}")
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Processing completed successfully!")
            
            # Save results
            output_file = str(Path(input_folder).parent / "pulsehunter_results.json")
            success = pulsehunter_core.save_report(detections, output_file)
            
            message = f"Processing completed! Found {len(detections)} detections."
            if not success:
                message += " (Note: Online upload failed, but results saved locally)"
                
            self.processing_finished.emit(True, message, detections)
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.log_updated.emit(f"ERROR: {error_msg}")
            self.log_updated.emit(traceback.format_exc())
            self.processing_finished.emit(False, error_msg, [])
            
    def cancel(self):
        """Cancel processing"""
        self.mutex.lock()
        self.is_cancelled = True
        self.mutex.unlock()
        self.status_updated.emit("Cancelling processing...")

class DetectionResultsWidget(QWidget):
    """Widget for displaying detection results"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detections = []
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls section
        controls_group = QGroupBox("Detection Filters")
        controls_layout = QHBoxLayout(controls_group)
        
        # Confidence filter
        controls_layout.addWidget(QLabel("Min Confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.filter_detections)
        controls_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("50%")
        controls_layout.addWidget(self.confidence_label)
        
        # Type filter
        controls_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["All", "Brightening", "Dimming", "GAIA Matched", "Exoplanet Candidates"])
        self.type_combo.currentTextChanged.connect(self.filter_detections)
        controls_layout.addWidget(self.type_combo)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.results_table)
        
        # Details section
        details_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Detection details
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(200)
        self.details_text.setReadOnly(True)
        details_splitter.addWidget(self.details_text)
        
        # Light curve plot
        self.plot_widget = self.create_plot_widget()
        details_splitter.addWidget(self.plot_widget)
        
        layout.addWidget(details_splitter)
        
    def create_plot_widget(self):
        """Create matplotlib plot widget"""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        return plot_widget
        
    def update_detections(self, detections):
        """Update the detection results"""
        self.detections = detections
        if detections:
            self.populate_table()
        else:
            self.show_empty_state()
            
    def show_empty_state(self):
        """Show beautiful empty state when no detections"""
        self.results_table.setRowCount(0)
        self.details_text.setText("""
        <div style='text-align: center; padding: 40px; color: #a0aec0;'>
            <div style='font-size: 48px; margin-bottom: 20px;'>*</div>
            <div style='font-size: 18px; font-weight: 600; margin-bottom: 10px; color: #63b3ed;'>
                No Detection Results Yet
            </div>
            <div style='font-size: 14px; line-height: 1.6;'>
                Process your FITS images to search for:<br/>
                • Optical SETI signals<br/>
                • Exoplanet transits<br/>
                • Stellar variability<br/>
                • Transient phenomena
            </div>
            <div style='margin-top: 20px; font-size: 12px; font-style: italic; color: #718096;'>
                Every photon could hold the key to a cosmic discovery
            </div>
        </div>
        """)
        
        # Clear the plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, '* Awaiting Data *', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, color='#4299e1',
                fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
        
    def populate_table(self):
        """Populate the results table"""
        filtered_detections = self.get_filtered_detections()
        
        self.results_table.setRowCount(len(filtered_detections))
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Frame", "Filename", "Position", "RA/Dec", "Z-Score", 
            "Confidence", "Type", "Match"
        ])
        
        for row, detection in enumerate(filtered_detections):
            # Frame number
            self.results_table.setItem(row, 0, QTableWidgetItem(str(detection.get('frame', ''))))
            
            # Filename
            filename = Path(detection.get('filename', '')).name
            self.results_table.setItem(row, 1, QTableWidgetItem(filename))
            
            # Pixel position
            x, y = detection.get('x', 0), detection.get('y', 0)
            pos_text = f"({x}, {y})"
            self.results_table.setItem(row, 2, QTableWidgetItem(pos_text))
            
            # RA/Dec
            ra, dec = detection.get('ra_deg'), detection.get('dec_deg')
            if ra is not None and dec is not None:
                coord_text = f"{ra:.4f}, {dec:.4f}"
            else:
                coord_text = "No WCS"
            self.results_table.setItem(row, 3, QTableWidgetItem(coord_text))
            
            # Z-score
            z_score = detection.get('z_score', 0)
            z_item = QTableWidgetItem(f"{z_score:.2f}")
            if abs(z_score) > 10:
                z_item.setBackground(QColor(255, 200, 200))  # Highlight high significance
            self.results_table.setItem(row, 4, z_item)
            
            # Confidence
            confidence = detection.get('confidence', 0)
            conf_text = f"{int(confidence * 100)}%"
            conf_item = QTableWidgetItem(conf_text)
            if confidence > 0.8:
                conf_item.setBackground(QColor(200, 255, 200))  # Green for high confidence
            elif confidence < 0.3:
                conf_item.setBackground(QColor(255, 200, 200))  # Red for low confidence
            self.results_table.setItem(row, 5, conf_item)
            
            # Type
            det_type = "Dimming" if detection.get('dimming') else "Brightening"
            if detection.get('exo_match'):
                det_type += " (Exo)"
            type_item = QTableWidgetItem(det_type)
            if detection.get('exo_match'):
                type_item.setBackground(QColor(200, 200, 255))  # Blue for exoplanet candidates
            self.results_table.setItem(row, 6, type_item)
            
            # Match info
            match_name = detection.get('match_name', 'Unmatched')
            if match_name and 'GAIA' in match_name:
                match_text = f"GAIA ({detection.get('angular_distance_arcsec', 0):.1f}\")"
            else:
                match_text = match_name or "None"
            self.results_table.setItem(row, 7, QTableWidgetItem(match_text))
        
        # Resize columns to content
        self.results_table.resizeColumnsToContents()
        
    def get_filtered_detections(self):
        """Get detections filtered by current criteria"""
        min_confidence = self.confidence_slider.value() / 100.0
        filter_type = self.type_combo.currentText()
        
        filtered = []
        for detection in self.detections:
            # Confidence filter
            if detection.get('confidence', 0) < min_confidence:
                continue
                
            # Type filter
            if filter_type == "Brightening" and detection.get('dimming'):
                continue
            elif filter_type == "Dimming" and not detection.get('dimming'):
                continue
            elif filter_type == "GAIA Matched" and not detection.get('match_name'):
                continue
            elif filter_type == "Exoplanet Candidates" and not detection.get('exo_match'):
                continue
                
            filtered.append(detection)
            
        return filtered
        
    def filter_detections(self):
        """Apply filters and update display"""
        self.confidence_label.setText(f"{self.confidence_slider.value()}%")
        self.populate_table()
        
    def on_selection_changed(self):
        """Handle table selection change"""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            filtered_detections = self.get_filtered_detections()
            if current_row < len(filtered_detections):
                detection = filtered_detections[current_row]
                self.show_detection_details(detection)
                
    def show_detection_details(self, detection):
        """Show detailed information for selected detection"""
        # Update details text
        details = f"""
Detection Details:
Frame: {detection.get('frame', 'N/A')}
File: {detection.get('filename', 'N/A')}
Position: ({detection.get('x', 0)}, {detection.get('y', 0)})
Z-Score: {detection.get('z_score', 0):.3f}
Confidence: {int(detection.get('confidence', 0) * 100)}%
Type: {'Dimming' if detection.get('dimming') else 'Brightening'}

Coordinates:
RA: {detection.get('ra_deg', 'N/A')}°
Dec: {detection.get('dec_deg', 'N/A')}°

Catalog Matches:
GAIA: {detection.get('match_name', 'None')}
        """
        
        if detection.get('g_mag'):
            details += f"G Magnitude: {detection.get('g_mag'):.2f}\n"
            
        if detection.get('exo_match'):
            exo = detection['exo_match']
            details += f"\nExoplanet Match:\n"
            details += f"Host: {exo.get('host', 'N/A')}\n"
            details += f"Planet: {exo.get('planet', 'N/A')}\n"
            details += f"Period: {exo.get('period_days', 'N/A')} days\n"
            details += f"Depth: {exo.get('depth_ppm', 'N/A')} ppm\n"
            
        self.details_text.setText(details.strip())
        
        # Plot light curve if available
        if detection.get('light_curve'):
            self.plot_light_curve(detection)
            
    def plot_light_curve(self, detection):
        """Plot light curve for selected detection"""
        light_curve = detection.get('light_curve', [])
        if not light_curve:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        frames = list(range(len(light_curve)))
        ax.plot(frames, light_curve, 'b-o', markersize=4, linewidth=1.5)
        
        # Highlight the detection frame
        det_frame = detection.get('frame', 0)
        if det_frame < len(light_curve):
            ax.plot(det_frame, light_curve[det_frame], 'ro', markersize=8, label='Detection')
            
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Brightness (ADU)')
        ax.set_title(f'Light Curve - {Path(detection.get("filename", "")).name}')
        ax.grid(True, alpha=0.3)
        
        if det_frame < len(light_curve):
            ax.legend()
            
        self.figure.tight_layout()
        self.canvas.draw()

class ProjectConfigDialog(QDialog):
    """Dialog for configuring processing parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Process Images - Configuration")
        self.setModal(True)
        self.resize(600, 500)
        self.project_data = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Input folder selection
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout(input_group)
        
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("FITS Folder:"))
        self.folder_edit = QFileDialog()
        self.folder_path = ""
        self.folder_label = QLabel("No folder selected")
        folder_layout.addWidget(self.folder_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_btn)
        input_layout.addLayout(folder_layout)
        
        layout.addWidget(input_group)
        
        # Processing parameters
        params_group = QGroupBox("Processing Parameters")
        params_layout = QGridLayout(params_group)
        
        # Detection threshold
        params_layout.addWidget(QLabel("Detection Threshold (σ):"), 0, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(3.0)
        self.threshold_spin.setMaximum(15.0)
        self.threshold_spin.setValue(6.0)
        self.threshold_spin.setSingleStep(0.5)
        params_layout.addWidget(self.threshold_spin, 0, 1)
        
        # Edge margin
        params_layout.addWidget(QLabel("Edge Margin (pixels):"), 1, 0)
        self.margin_spin = QSpinBox()
        self.margin_spin.setMinimum(5)
        self.margin_spin.setMaximum(100)
        self.margin_spin.setValue(20)
        params_layout.addWidget(self.margin_spin, 1, 1)
        
        # Cutout size
        params_layout.addWidget(QLabel("Cutout Size (pixels):"), 2, 0)
        self.cutout_spin = QSpinBox()
        self.cutout_spin.setMinimum(20)
        self.cutout_spin.setMaximum(200)
        self.cutout_spin.setValue(50)
        params_layout.addWidget(self.cutout_spin, 2, 1)
        
        # Plate solving
        self.plate_solve_check = QCheckBox("Enable plate solving for files without WCS")
        self.plate_solve_check.setChecked(True)
        params_layout.addWidget(self.plate_solve_check, 3, 0, 1, 2)
        
        # GAIA cross-matching
        self.gaia_check = QCheckBox("Cross-match with GAIA DR3 catalog")
        self.gaia_check.setChecked(True)
        params_layout.addWidget(self.gaia_check, 4, 0, 1, 2)
        
        # Exoplanet matching
        self.exo_check = QCheckBox("Cross-match with NASA Exoplanet Archive")
        self.exo_check.setChecked(True)
        params_layout.addWidget(self.exo_check, 5, 0, 1, 2)
        
        layout.addWidget(params_group)
        
        # Calibration status
        cal_group = QGroupBox("Calibration Status")
        cal_layout = QVBoxLayout(cal_group)
        
        self.cal_status_label = QLabel("No calibration configured")
        self.cal_status_label.setStyleSheet("color: #666;")
        cal_layout.addWidget(self.cal_status_label)
        
        cal_btn = QPushButton("Setup Calibration...")
        cal_btn.clicked.connect(self.setup_calibration)
        cal_layout.addWidget(cal_btn)
        
        layout.addWidget(cal_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        start_btn = QPushButton("🚀 Start Processing")
        start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
                color: white;
                border: none;
                padding: 12px 24px;
                font-weight: bold;
                border-radius: 8px;
                font-size: 14px;
                min-height: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68d391, stop:1 #48bb78);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #38a169, stop:1 #2f855a);
            }
        """)
        start_btn.clicked.connect(self.accept)
        button_layout.addWidget(start_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def browse_folder(self):
        """Browse for input folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select FITS Images Folder", ""
        )
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"Selected: {Path(folder).name}")
            
            # Count FITS files
            fits_count = len(list(Path(folder).glob("*.fit*")))
            if fits_count > 0:
                self.folder_label.setText(f"Selected: {Path(folder).name} ({fits_count} FITS files)")
            else:
                self.folder_label.setText(f"Selected: {Path(folder).name} (No FITS files found!)")
                self.folder_label.setStyleSheet("color: red;")
                
    def setup_calibration(self):
        """Open calibration dialog"""
        dialog = CalibrationSetupDialog(self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            self.cal_status_label.setText("Calibration configured ✓")
            self.cal_status_label.setStyleSheet("color: green;")
            
    def get_project_data(self):
        """Get configured project data"""
        if not self.folder_path:
            return None
            
        return {
            'input_folder': self.folder_path,
            'detection_threshold': self.threshold_spin.value(),
            'edge_margin': self.margin_spin.value(),
            'cutout_size': self.cutout_spin.value(),
            'plate_solve': self.plate_solve_check.isChecked(),
            'gaia_match': self.gaia_check.isChecked(),
            'exo_match': self.exo_check.isChecked(),
            'astap_path': 'astap',  # Will be updated from config
            'master_bias': None,
            'master_dark': None,
            'master_flat': None
        }

class PulseHunterMainWindow(QMainWindow):
    """Enhanced PulseHunter main window - Fully Functional"""

    def __init__(self):
        super().__init__()

        # Initialize core components
        self.settings = QSettings('PulseHunter', 'MainApplication')
        self.config = CalibrationConfig()
        self.astap_manager = ASTAPManager(self.config)
        self.logger = CalibrationLogger()
        
        # Processing state
        self.processing_worker = None
        self.current_detections = []
        self.current_project = {}

        # UI components
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
        self.setup_central_widget()

        # Initialize systems
        self.initialize_astap()
        self.restore_geometry()

        # Log application startup
        self.logger.info("PulseHunter application started")

    def setup_ui(self):
        """Setup main UI properties"""
        self.setWindowTitle("PulseHunter - Optical SETI & Exoplanet Detection Pipeline")
        self.setMinimumSize(1400, 900)

        # Set application icon if available
        icon_path = Path("resources/icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        # New project
        new_action = QAction('&New Project...', self)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('Create a new observation project')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        # Open project
        open_action = QAction('&Open Project...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open an existing project')
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # Save project
        save_action = QAction('&Save Project...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save current project')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit PulseHunter')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Calibration menu (enhanced)
        calibration_menu = menubar.addMenu('&Calibration')

        # Main calibration setup
        setup_action = QAction('&Calibration Setup...', self)
        setup_action.setShortcut('Ctrl+Shift+C')
        setup_action.setStatusTip('Configure and create master calibration files')
        setup_action.triggered.connect(self.open_calibration_dialog)
        calibration_menu.addAction(setup_action)

        calibration_menu.addSeparator()

        # ASTAP configuration
        astap_config_action = QAction('Configure &ASTAP...', self)
        astap_config_action.setStatusTip('Configure ASTAP plate solving executable')
        astap_config_action.triggered.connect(self.configure_astap)
        calibration_menu.addAction(astap_config_action)

        # Test ASTAP
        test_astap_action = QAction('&Test ASTAP Connection', self)
        test_astap_action.setStatusTip('Test ASTAP executable')
        test_astap_action.triggered.connect(self.test_astap)
        calibration_menu.addAction(test_astap_action)

        # Processing menu
        processing_menu = menubar.addMenu('&Processing')

        process_images_action = QAction('&Process Images...', self)
        process_images_action.setShortcut('Ctrl+P')
        process_images_action.setStatusTip('Process FITS images for detection')
        process_images_action.triggered.connect(self.process_images)
        processing_menu.addAction(process_images_action)

        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')

        view_results_action = QAction('&View Detection Results...', self)
        view_results_action.setShortcut('Ctrl+R')
        view_results_action.setStatusTip('View analysis results and detections')
        view_results_action.triggered.connect(self.view_results)
        analysis_menu.addAction(view_results_action)
        
        # Export menu
        export_menu = analysis_menu.addMenu('&Export Results')
        
        export_csv_action = QAction('Export to &CSV...', self)
        export_csv_action.triggered.connect(self.export_csv)
        export_menu.addAction(export_csv_action)
        
        export_json_action = QAction('Export to &JSON...', self)
        export_json_action.triggered.connect(self.export_json)
        export_menu.addAction(export_json_action)

        # Tools menu
        tools_menu = menubar.addMenu('&Tools')

        preferences_action = QAction('&Preferences...', self)
        preferences_action.setStatusTip('Configure application preferences')
        preferences_action.triggered.connect(self.open_preferences)
        tools_menu.addAction(preferences_action)

        # Help menu
        help_menu = menubar.addMenu('&Help')

        documentation_action = QAction('&Documentation', self)
        documentation_action.setStatusTip('Open PulseHunter documentation')
        documentation_action.triggered.connect(self.open_documentation)
        help_menu.addAction(documentation_action)

        help_menu.addSeparator()

        about_action = QAction('&About PulseHunter', self)
        about_action.setStatusTip('About PulseHunter')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup status bar with system status indicators"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Main status message
        self.status_bar.showMessage("Ready")

        # ASTAP status indicator
        self.astap_status_label = QLabel("ASTAP: Not configured")
        self.astap_status_label.setStyleSheet("color: #666; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.astap_status_label)

        # Detection count
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setStyleSheet("color: #666; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.detection_count_label)

        # Update status periodically
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_indicators)
        self.status_timer.start(30000)  # Update every 30 seconds

    def setup_central_widget(self):
        """Setup the main central widget area"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Project overview tab
        self.setup_project_tab()

        # Processing tab
        self.setup_processing_tab()

        # Results tab
        self.setup_results_tab()

        # Log tab
        self.setup_log_tab()

    def setup_project_tab(self):
        """Setup project overview tab"""
        project_widget = QWidget()
        layout = QVBoxLayout(project_widget)

        # Welcome section
        welcome_group = QGroupBox(">> Welcome to PulseHunter <<")
        welcome_layout = QVBoxLayout(welcome_group)

        welcome_text = QLabel("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #4299e1; margin-bottom: 10px; font-size: 28px;'>
                🌌 PulseHunter 🌌
            </h1>
            <h2 style='color: #63b3ed; margin-bottom: 20px; font-size: 18px; font-weight: normal;'>
                Optical SETI & Exoplanet Detection Pipeline
            </h2>
            
            <p style='color: #e8e8e8; font-size: 16px; margin-bottom: 25px; line-height: 1.6;'>
                Welcome to <strong>PulseHunter</strong>, your gateway to citizen science astronomy!<br/>
                <em>Join the global network of citizen scientists searching for extraterrestrial intelligence and discovering new exoplanets.</em>
            </p>
        </div>
        
        <div style='background: linear-gradient(135deg, rgba(66, 153, 225, 0.1), rgba(49, 130, 206, 0.05)); 
                    border: 1px solid #4299e1; border-radius: 12px; padding: 20px; margin: 10px;'>
            <h3 style='color: #4299e1; margin-bottom: 15px; font-size: 18px;'>
                🚀 Getting Started:
            </h3>
            <ol style='color: #e8e8e8; font-size: 14px; line-height: 1.8; padding-left: 20px;'>
                <li><strong style='color: #48bb78;'>Configure ASTAP:</strong> Set up plate solving (Calibration → Configure ASTAP)</li>
                <li><strong style='color: #4299e1;'>Setup Calibration:</strong> Create master calibration files (Calibration → Calibration Setup)</li>
                <li><strong style='color: #ed8936;'>Process Images:</strong> Analyze your FITS files for detections (Processing → Process Images)</li>
                <li><strong style='color: #f56565;'>Review Results:</strong> Examine potential discoveries (Analysis → View Results)</li>
            </ol>
        </div>
        
        <div style='text-align: center; margin-top: 20px; padding: 15px;'>
            <p style='color: #a0aec0; font-size: 13px; margin-bottom: 10px;'>
                📖 <a href="https://geekastro.dev/pulsehunter/" style='color: #4299e1; text-decoration: none;'>
                    Visit the PulseHunter website
                </a> for documentation, tutorials, and community updates.
            </p>
            <p style='color: #63b3ed; font-size: 12px; font-style: italic;'>
                * Every detection could be the discovery that changes everything *
            </p>
        </div>
        """)
        welcome_text.setWordWrap(True)
        welcome_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        welcome_layout.addWidget(welcome_text)

        layout.addWidget(welcome_group)

        # Quick actions
        actions_group = QGroupBox(">> Quick Actions")
        actions_layout = QHBoxLayout(actions_group)

        calibration_btn = QPushButton("🔭 Setup Calibration")
        calibration_btn.setProperty("class", "success")
        calibration_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
                color: white;
                border: none;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 12px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68d391, stop:1 #48bb78);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #38a169, stop:1 #2f855a);
            }
        """)
        calibration_btn.clicked.connect(self.open_calibration_dialog)
        actions_layout.addWidget(calibration_btn)

        process_btn = QPushButton("🚀 Process Images")
        process_btn.setProperty("class", "primary")
        process_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299e1, stop:1 #3182ce);
                color: white;
                border: none;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 12px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #63b3ed, stop:1 #4299e1);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3182ce, stop:1 #2c5aa0);
            }
        """)
        process_btn.clicked.connect(self.process_images)
        actions_layout.addWidget(process_btn)

        results_btn = QPushButton("📊 View Results")
        results_btn.setProperty("class", "warning")
        results_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ed8936, stop:1 #dd6b20);
                color: white;
                border: none;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 12px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f6ad55, stop:1 #ed8936);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #dd6b20, stop:1 #c05621);
            }
        """)
        results_btn.clicked.connect(self.view_results)
        actions_layout.addWidget(results_btn)

        actions_layout.addStretch()
        layout.addWidget(actions_group)
        
        # Project status
        status_group = QGroupBox(">> Current Project Status")
        status_layout = QVBoxLayout(status_group)
        
        self.project_status_label = QLabel("""
        <div style='text-align: center; padding: 20px;'>
            <div style='color: #a0aec0; font-size: 16px; margin-bottom: 10px;'>
                * No project loaded
            </div>
            <div style='color: #718096; font-size: 13px; font-style: italic;'>
                Create a new project or open an existing one to begin your cosmic search
            </div>
        </div>
        """)
        self.project_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.project_status_label)
        
        layout.addWidget(status_group)

        layout.addStretch()
        self.tab_widget.addTab(project_widget, "Project")

    def setup_processing_tab(self):
        """Setup processing tab"""
        processing_widget = QWidget()
        layout = QVBoxLayout(processing_widget)

        # Processing controls
        controls_group = QGroupBox("Processing Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Progress bar
        self.processing_progress = QProgressBar()
        self.processing_progress.setVisible(False)
        controls_layout.addWidget(self.processing_progress)

        # Status
        self.processing_status = QLabel("No processing active")
        controls_layout.addWidget(self.processing_status)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("⏹️ Cancel Processing")
        self.cancel_btn.setProperty("class", "danger")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f56565, stop:1 #e53e3e);
                color: white;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #fc8181, stop:1 #f56565);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e53e3e, stop:1 #c53030);
            }
        """)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)

        layout.addWidget(controls_group)

        # Processing log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.processing_log = QTextEdit()
        self.processing_log.setFont(QFont("Consolas", 10))
        self.processing_log.setReadOnly(True)
        log_layout.addWidget(self.processing_log)

        layout.addWidget(log_group)

        self.tab_widget.addTab(processing_widget, "Processing")

    def setup_results_tab(self):
        """Setup results tab with detection results widget"""
        results_container = QWidget()
        container_layout = QVBoxLayout(results_container)
        
        # Results header
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        title_label = QLabel(">> Detection Results")
        title_label.setStyleSheet("""
            QLabel {
                color: #4299e1;
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Export buttons
        export_csv_btn = QPushButton("📊 Export CSV")
        export_csv_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: 600;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68d391, stop:1 #48bb78);
            }
        """)
        export_csv_btn.clicked.connect(self.export_csv)
        header_layout.addWidget(export_csv_btn)
        
        export_json_btn = QPushButton("💾 Export JSON")
        export_json_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299e1, stop:1 #3182ce);
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: 600;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #63b3ed, stop:1 #4299e1);
            }
        """)
        export_json_btn.clicked.connect(self.export_json)
        header_layout.addWidget(export_json_btn)
        
        container_layout.addWidget(header_widget)
        
        self.results_widget = DetectionResultsWidget()
        container_layout.addWidget(self.results_widget)
        
        self.tab_widget.addTab(results_container, "Results")

    def setup_log_tab(self):
        """Setup system log tab"""
        log_widget = QWidget()
        layout = QVBoxLayout(log_widget)

        # System log
        self.system_log = QTextEdit()
        self.system_log.setFont(QFont("Consolas", 9))
        self.system_log.setReadOnly(True)
        layout.addWidget(self.system_log)

        # Add initial log entries
        self.add_system_log("PulseHunter application started")
        self.add_system_log("Checking system configuration...")

        self.tab_widget.addTab(log_widget, "System Log")

    def initialize_astap(self):
        """Initialize ASTAP configuration on startup"""
        self.logger.info("Initializing ASTAP configuration...")
        self.add_system_log("Initializing ASTAP configuration...")

        # Auto-detect if enabled and not already configured
        if (self.config.getboolean('ASTAP', 'auto_detect_on_startup', True) and
            not self.astap_manager.astap_path):

            self.add_system_log("Auto-detecting ASTAP executable...")
            detected_path = self.astap_manager.auto_detect_astap()
            if detected_path:
                self.logger.info(f"Auto-detected ASTAP at startup: {detected_path}")
                self.add_system_log(f"ASTAP auto-detected: {Path(detected_path).name}")
            else:
                self.add_system_log("ASTAP not found - manual configuration required")

        self.update_astap_status()

    def update_status_indicators(self):
        """Update all status indicators"""
        self.update_astap_status()
        self.update_detection_count()

    def update_astap_status(self):
        """Update ASTAP status in status bar"""
        status_info = self.astap_manager.get_status_info()

        if status_info['valid']:
            self.astap_status_label.setText(f"ASTAP: Ready ({Path(status_info['path']).name})")
            self.astap_status_label.setStyleSheet("color: green; padding: 0 10px;")
            self.astap_status_label.setToolTip(f"ASTAP Path: {status_info['path']}\n{status_info['version']}")
        elif status_info['configured']:
            self.astap_status_label.setText("ASTAP: Configuration error")
            self.astap_status_label.setStyleSheet("color: red; padding: 0 10px;")
            self.astap_status_label.setToolTip(f"ASTAP Path: {status_info['path']}\nError: {status_info['message']}")
        else:
            self.astap_status_label.setText("ASTAP: Not configured")
            self.astap_status_label.setStyleSheet("color: #666; padding: 0 10px;")
            self.astap_status_label.setToolTip("ASTAP executable not configured. Use Calibration > Configure ASTAP")
            
    def update_detection_count(self):
        """Update detection count in status bar"""
        count = len(self.current_detections)
        self.detection_count_label.setText(f"Detections: {count}")
        
        if count > 0:
            high_conf = sum(1 for d in self.current_detections if d.get('confidence', 0) > 0.8)
            if high_conf > 0:
                self.detection_count_label.setStyleSheet("color: green; padding: 0 10px; font-weight: bold;")
                self.detection_count_label.setToolTip(f"Total: {count} detections\nHigh confidence: {high_conf}")
            else:
                self.detection_count_label.setStyleSheet("color: orange; padding: 0 10px;")
                self.detection_count_label.setToolTip(f"Total: {count} detections\nNo high confidence detections")
        else:
            self.detection_count_label.setStyleSheet("color: #666; padding: 0 10px;")

    def add_system_log(self, message):
        """Add message to system log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.system_log.append(f"[{timestamp}] {message}")
        
    def add_processing_log(self, message):
        """Add message to processing log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.processing_log.append(f"[{timestamp}] {message}")

    # Menu action handlers
    def new_project(self):
        """Create new project"""
        self.current_project = {
            'created': datetime.now().isoformat(),
            'detections': [],
            'parameters': {}
        }
        self.current_detections = []
        self.results_widget.update_detections([])
        self.update_project_status()
        self.add_system_log("New project created")

    def open_project(self):
        """Open existing project"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PulseHunter Project", "",
            "PulseHunter Projects (*.phproj);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    if file_path.endswith('.phproj'):
                        project_data = pickle.load(f)
                    else:
                        project_data = json.load(f)
                
                self.current_project = project_data
                self.current_detections = project_data.get('detections', [])
                self.results_widget.update_detections(self.current_detections)
                self.update_project_status()
                self.add_system_log(f"Project loaded: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Project", f"Could not load project:\n{str(e)}")
                
    import numpy as np

    def save_project(self):
        """Save current project"""
        if not self.current_project:
            QMessageBox.information(self, "No Project", "No project to save. Create a new project first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PulseHunter Project", "",
            "PulseHunter Projects (*.phproj);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                self.current_project['detections'] = self.current_detections
                self.current_project['saved'] = datetime.now().isoformat()

                def convert_numpy(obj):
                    if isinstance(obj, np.generic):
                        return obj.item()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

                with open(file_path, 'w') as f:
                    if file_path.endswith('.phproj'):
                        pickle.dump(self.current_project, f)
                    else:
                        json.dump(self.current_project, f, indent=2, default=convert_numpy)

                self.add_system_log(f"Project saved: {Path(file_path).name}")

            except Exception as e:
                QMessageBox.critical(self, "Error Saving Project", f"Could not save project:\n{str(e)}")

    def update_project_status(self):
        """Update project status display"""
        if self.current_project:
            created = self.current_project.get('created', 'Unknown')
            det_count = len(self.current_detections)
            high_conf = sum(1 for d in self.current_detections if d.get('confidence', 0) > 0.8)
            exo_candidates = sum(1 for d in self.current_detections if d.get('exo_match'))
            
            # Format the creation time nicely
            try:
                from datetime import datetime
                created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                created_str = created_dt.strftime("%B %d, %Y at %H:%M")
            except:
                created_str = created
            
            status_html = f"""
            <div style='text-align: center; padding: 20px;'>
                <div style='color: #4299e1; font-size: 18px; font-weight: bold; margin-bottom: 15px;'>
                    ✅ Project Active
                </div>
                
                <div style='background: linear-gradient(135deg, rgba(66, 153, 225, 0.1), rgba(49, 130, 206, 0.05)); 
                            border: 1px solid #4299e1; border-radius: 8px; padding: 15px; margin: 10px 0;'>
                    <div style='color: #e8e8e8; font-size: 14px; line-height: 1.6;'>
                        <div style='margin-bottom: 8px;'>
                            <strong style='color: #63b3ed;'>Created:</strong> {created_str}
                        </div>
                        <div style='margin-bottom: 8px;'>
                            <strong style='color: #63b3ed;'>Total Detections:</strong> 
                            <span style='color: #48bb78; font-weight: bold;'>{det_count}</span>
                        </div>
                        <div style='margin-bottom: 8px;'>
                            <strong style='color: #63b3ed;'>High Confidence:</strong> 
                            <span style='color: #f6ad55; font-weight: bold;'>{high_conf}</span>
                        </div>
                        <div>
                            <strong style='color: #63b3ed;'>Exoplanet Candidates:</strong> 
                            <span style='color: #fc8181; font-weight: bold;'>{exo_candidates}</span>
                        </div>
                    </div>
                </div>
                
                <div style='color: #a0aec0; font-size: 12px; font-style: italic; margin-top: 10px;'>
                    * Ready for analysis and discovery *
                </div>
            </div>
            """
            self.project_status_label.setText(status_html)
        else:
            self.project_status_label.setText("""
            <div style='text-align: center; padding: 20px;'>
                <div style='color: #a0aec0; font-size: 16px; margin-bottom: 10px;'>
                    * No project loaded
                </div>
                <div style='color: #718096; font-size: 13px; font-style: italic;'>
                    Create a new project or open an existing one to begin your cosmic search
                </div>
            </div>
            """)

    def open_calibration_dialog(self):
        """Open the enhanced calibration setup dialog"""
        try:
            self.add_system_log("Opening calibration setup dialog...")
            dialog = CalibrationSetupDialog(self)
            result = dialog.exec()

            # Update ASTAP status after dialog closes
            self.update_astap_status()

            if result == QDialog.DialogCode.Accepted:
                self.add_system_log("Calibration setup completed")
            else:
                self.add_system_log("Calibration setup cancelled")

        except Exception as e:
            error_msg = f"Error opening calibration dialog: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(
                self,
                "Calibration Dialog Error",
                f"Could not open calibration dialog:\n\n{error_msg}"
            )

    def configure_astap(self):
        """Quick ASTAP configuration dialog"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Configure ASTAP")
        dialog.setModal(True)
        dialog.resize(600, 250)

        layout = QVBoxLayout(dialog)

        # Instructions
        instructions = QLabel(
            "ASTAP is required for plate solving and astrometric calibration.\n"
            "Please specify the location of your ASTAP executable."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Current path display
        current_path_label = QLabel("Current ASTAP Path:")
        layout.addWidget(current_path_label)

        path_edit = QLineEdit(self.astap_manager.astap_path)
        path_edit.setReadOnly(True)
        path_edit.setStyleSheet("background-color: #f5f5f5;")
        layout.addWidget(path_edit)

        # Browse buttons
        browse_layout = QHBoxLayout()

        browse_btn = QPushButton("Browse for ASTAP Executable...")
        def browse_astap():
            file_path, _ = QFileDialog.getOpenFileName(
                dialog,
                "Select ASTAP Executable",
                "",
                "Executable Files (*.exe);;All Files (*)" if sys.platform == "win32" else "All Files (*)"
            )
            if file_path:
                path_edit.setText(file_path)
                if self.astap_manager.validate_astap_executable(file_path):
                    self.astap_manager.astap_path = file_path
                    self.add_system_log(f"ASTAP configured: {Path(file_path).name}")
                    QMessageBox.information(dialog, "Success", "ASTAP configured successfully!")
                else:
                    QMessageBox.warning(dialog, "Validation Failed", "Selected file failed ASTAP validation.")

        browse_btn.clicked.connect(browse_astap)
        browse_layout.addWidget(browse_btn)

        auto_detect_btn = QPushButton("Auto-Detect")
        def auto_detect():
            detected = self.astap_manager.auto_detect_astap()
            if detected:
                path_edit.setText(detected)
                self.add_system_log(f"ASTAP auto-detected: {Path(detected).name}")
                QMessageBox.information(dialog, "Success", f"ASTAP auto-detected at:\n{detected}")
            else:
                QMessageBox.information(dialog, "Not Found", "Could not auto-detect ASTAP executable.")
        auto_detect_btn.clicked.connect(auto_detect)
        browse_layout.addWidget(auto_detect_btn)

        browse_layout.addStretch()
        layout.addLayout(browse_layout)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()
        self.update_astap_status()

    def test_astap(self):
        """Test ASTAP connection"""
        if not self.astap_manager.astap_path:
            QMessageBox.warning(
                self,
                "ASTAP Not Configured",
                "ASTAP executable is not configured.\n\nUse 'Configure ASTAP...' to set the executable path."
            )
            return

        try:
            self.add_system_log("Testing ASTAP connection...")
            version_info = self.astap_manager.get_astap_version()
            self.add_system_log("ASTAP test successful")

            QMessageBox.information(
                self,
                "ASTAP Test Successful",
                f"ASTAP is working correctly!\n\n"
                f"Executable: {Path(self.astap_manager.astap_path).name}\n"
                f"Location: {self.astap_manager.astap_path}\n"
                f"Version: {version_info}"
            )
        except Exception as e:
            error_msg = f"ASTAP test failed: {str(e)}"
            self.add_system_log(error_msg)
            QMessageBox.critical(
                self,
                "ASTAP Test Failed",
                f"ASTAP test failed:\n\n{error_msg}\n\n"
                f"Please check the executable path and ensure ASTAP is properly installed."
            )

    def process_images(self):
        """Process FITS images - Now fully functional"""
        self.add_system_log("Starting image processing configuration...")
        
        # Open configuration dialog
        config_dialog = ProjectConfigDialog(self)
        result = config_dialog.exec()
        
        if result != QDialog.DialogCode.Accepted:
            return
            
        project_data = config_dialog.get_project_data()
        if not project_data:
            QMessageBox.warning(self, "Configuration Error", "Please select a FITS folder to process.")
            return
            
        # Update with ASTAP path
        project_data['astap_path'] = self.astap_manager.astap_path or 'astap'
        
        # Start processing in worker thread
        self.processing_worker = ProcessingWorker(project_data)
        self.processing_worker.progress_updated.connect(self.update_processing_progress)
        self.processing_worker.status_updated.connect(self.update_processing_status)
        self.processing_worker.log_updated.connect(self.add_processing_log)
        self.processing_worker.processing_finished.connect(self.processing_finished)
        
        # Update UI for processing
        self.processing_progress.setVisible(True)
        self.processing_progress.setValue(0)
        self.cancel_btn.setVisible(True)
        self.processing_status.setText("Starting processing...")
        
        # Switch to processing tab
        self.tab_widget.setCurrentIndex(1)
        
        # Start processing
        self.processing_worker.start()
        self.add_system_log("Image processing started")
        
    def update_processing_progress(self, value):
        """Update processing progress bar"""
        self.processing_progress.setValue(value)
        
    def update_processing_status(self, message):
        """Update processing status"""
        self.processing_status.setText(message)
        
    def cancel_processing(self):
        """Cancel current processing"""
        if self.processing_worker:
            self.processing_worker.cancel()
            self.add_processing_log("Processing cancellation requested...")
            
    def processing_finished(self, success, message, detections):
        """Handle processing completion"""
        # Update UI
        self.processing_progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        
        if success:
            self.processing_status.setText("Processing completed successfully!")
            self.current_detections = detections
            self.results_widget.update_detections(detections)
            
            # Update project
            if not self.current_project:
                self.current_project = {'created': datetime.now().isoformat()}
            self.current_project['detections'] = detections
            self.current_project['last_processed'] = datetime.now().isoformat()
            
            self.update_project_status()
            
            # Switch to results tab
            self.tab_widget.setCurrentIndex(2)
            
            QMessageBox.information(self, "Processing Complete", message)
        else:
            self.processing_status.setText("Processing failed!")
            QMessageBox.critical(self, "Processing Error", f"Processing failed:\n\n{message}")
            
        self.processing_worker = None
        self.update_status_indicators()

    def view_results(self):
        """View analysis results - Now fully functional"""
        if not self.current_detections:
            QMessageBox.information(
                self, "No Results", 
                "No detection results available.\n\nUse Processing → Process Images to analyze your FITS files."
            )
            return
            
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)
        self.add_system_log("Viewing detection results")
        
    def export_csv(self):
        """Export results to CSV"""
        if not self.current_detections:
            QMessageBox.information(self, "No Data", "No detection results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Detection Results", "pulsehunter_detections.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        'Frame', 'Filename', 'X', 'Y', 'RA_deg', 'Dec_deg', 
                        'Z_Score', 'Confidence', 'Dimming', 'GAIA_Match', 
                        'G_Magnitude', 'Exoplanet_Match'
                    ])
                    
                    # Data rows
                    for det in self.current_detections:
                        writer.writerow([
                            det.get('frame', ''),
                            det.get('filename', ''),
                            det.get('x', ''),
                            det.get('y', ''),
                            det.get('ra_deg', ''),
                            det.get('dec_deg', ''),
                            det.get('z_score', ''),
                            det.get('confidence', ''),
                            det.get('dimming', ''),
                            det.get('match_name', ''),
                            det.get('g_mag', ''),
                            det.get('exo_match', {}).get('planet', '') if det.get('exo_match') else ''
                        ])
                        
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
                self.add_system_log(f"Results exported to CSV: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export results:\n{str(e)}")
                
    def export_json(self):
        """Export results to JSON"""
        if not self.current_detections:
            QMessageBox.information(self, "No Data", "No detection results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Detection Results", "pulsehunter_detections.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                export_data = {
                    'metadata': {
                        'exported': datetime.now().isoformat(),
                        'total_detections': len(self.current_detections),
                        'pulsehunter_version': 'Alpha-Enhanced'
                    },
                    'detections': self.current_detections
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
                self.add_system_log(f"Results exported to JSON: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export results:\n{str(e)}")

    def open_preferences(self):
        """Open preferences dialog"""
        QMessageBox.information(
            self,
            "Preferences",
            "Preferences dialog will be implemented here.\n\n"
            "This will include detection thresholds, output settings, "
            "network configuration, and other application preferences."
        )

    def open_documentation(self):
        """Open documentation"""
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl

        # Try to open PulseHunter documentation site
        QDesktopServices.openUrl(QUrl("https://geekastro.dev/pulsehunter/"))

    def show_about(self):
        """Show about dialog"""
        astap_status = "✓ Configured" if self.astap_manager.is_configured() else "✗ Not configured"
        det_count = len(self.current_detections)

        QMessageBox.about(
            self,
            "About PulseHunter",
            f"""
            <h3>PulseHunter</h3>
            <p><b>Optical SETI and Exoplanet Transit Detection Pipeline</b></p>
            <p>Version: Alpha (Enhanced & Fully Functional)</p>

            <p>PulseHunter empowers amateur astronomers worldwide to contribute
            to cutting-edge astronomical research through citizen science.</p>

            <p><b>System Status:</b></p>
            <p>ASTAP: {astap_status}</p>
            <p>Current Detections: {det_count}</p>

            <p><b>Features:</b></p>
            <ul>
            <li>Advanced calibration pipeline with FITS processing</li>
            <li>ASTAP plate solving integration</li>
            <li>Statistical transient detection algorithms</li>
            <li>GAIA DR3 catalog cross-matching</li>
            <li>NASA Exoplanet Archive integration</li>
            <li>Interactive results visualization</li>
            <li>Light curve analysis and plotting</li>
            <li>Project management and data export</li>
            <li>Real-time processing with progress tracking</li>
            </ul>

            <p>© 2025 Kelsi Davis - GeekAstro Development</p>
            <p><a href="https://geekastro.dev/pulsehunter/">https://geekastro.dev/pulsehunter/</a></p>
            <p><a href="https://github.com/Kelsidavis/PulseHunter">GitHub Repository</a></p>
            """
        )

    def restore_geometry(self):
        """Restore window geometry"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Center on screen
            screen = QApplication.primaryScreen().availableGeometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)

    def closeEvent(self, event):
        """Handle application closing"""
        # Cancel any running processing
        if self.processing_worker and self.processing_worker.isRunning():
            reply = QMessageBox.question(
                self, "Processing Active", 
                "Image processing is still running. Cancel processing and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.processing_worker.cancel()
                self.processing_worker.wait(3000)  # Wait up to 3 seconds
            else:
                event.ignore()
                return
                
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())

        # Log application shutdown
        self.logger.info("PulseHunter application closing")
        self.add_system_log("Application shutting down...")

        # Accept the close event
        event.accept()

def apply_dark_theme(app):
    """Apply beautiful dark space theme"""
    dark_stylesheet = """
    /* Main Application Style */
    QApplication {
        background-color: #0a0e1a;
        color: #e8e8e8;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    /* Main Window */
    QMainWindow {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #0a0e1a, stop:0.3 #1a1f2e, stop:0.7 #0f1419, stop:1 #0a0e1a);
        color: #e8e8e8;
        border: none;
    }
    
    /* Menu Bar */
    QMenuBar {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2d3748, stop:1 #1a202c);
        color: #e8e8e8;
        border-bottom: 2px solid #4299e1;
        padding: 4px;
        font-weight: 500;
    }
    
    QMenuBar::item {
        background: transparent;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 2px;
    }
    
    QMenuBar::item:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        color: white;
    }
    
    QMenu {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2d3748, stop:1 #1a202c);
        color: #e8e8e8;
        border: 1px solid #4299e1;
        border-radius: 8px;
        padding: 6px;
    }
    
    QMenu::item {
        padding: 8px 16px;
        border-radius: 4px;
        margin: 1px;
    }
    
    QMenu::item:selected {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4299e1, stop:1 #3182ce);
        color: white;
    }
    
    /* Status Bar */
    QStatusBar {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #1a202c, stop:1 #2d3748);
        color: #a0aec0;
        border-top: 1px solid #4299e1;
        padding: 4px;
    }
    
    QStatusBar QLabel {
        color: #a0aec0;
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    /* Tab Widget */
    QTabWidget::pane {
        border: 1px solid #4299e1;
        border-radius: 8px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #1a202c, stop:1 #0f1419);
        margin-top: 4px;
    }
    
    QTabBar::tab {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2d3748, stop:1 #1a202c);
        color: #a0aec0;
        padding: 12px 20px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        border: 1px solid #4a5568;
        font-weight: 500;
    }
    
    QTabBar::tab:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        color: white;
        border-color: #4299e1;
        font-weight: 600;
    }
    
    QTabBar::tab:hover:!selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a5568, stop:1 #2d3748);
        color: #e8e8e8;
    }
    
    /* Group Boxes */
    QGroupBox {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #1a202c, stop:1 #2d3748);
        border: 2px solid #4299e1;
        border-radius: 12px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: 600;
        color: #e8e8e8;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 16px;
        top: 6px;
        color: #4299e1;
        font-size: 14px;
        font-weight: 700;
        padding: 4px 8px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #1a202c, stop:0.5 #2d3748, stop:1 #1a202c);
        border-radius: 6px;
    }
    
    /* Buttons */
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 13px;
        min-height: 16px;
    }
    
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #63b3ed, stop:1 #4299e1);
    }
    
    QPushButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #3182ce, stop:1 #2c5aa0);
    }
    
    QPushButton:disabled {
        background: #4a5568;
        color: #a0aec0;
    }
    
    /* Special Button Styles */
    QPushButton[class="success"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #48bb78, stop:1 #38a169);
    }
    
    QPushButton[class="success"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #68d391, stop:1 #48bb78);
    }
    
    QPushButton[class="warning"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ed8936, stop:1 #dd6b20);
    }
    
    QPushButton[class="warning"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f6ad55, stop:1 #ed8936);
    }
    
    QPushButton[class="danger"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f56565, stop:1 #e53e3e);
    }
    
    QPushButton[class="danger"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #fc8181, stop:1 #f56565);
    }
    
    /* Text Fields */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background: #2d3748;
        border: 2px solid #4a5568;
        border-radius: 6px;
        padding: 8px;
        color: #e8e8e8;
        font-size: 13px;
        selection-background-color: #4299e1;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #4299e1;
        background: #374151;
    }
    
    /* Progress Bar */
    QProgressBar {
        background: #2d3748;
        border: 2px solid #4a5568;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: 600;
        height: 20px;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4299e1, stop:0.5 #63b3ed, stop:1 #4299e1);
        border-radius: 6px;
        margin: 2px;
    }
    
    /* Tables */
    QTableWidget {
        background: #1a202c;
        alternate-background-color: #2d3748;
        border: 1px solid #4299e1;
        border-radius: 8px;
        gridline-color: #4a5568;
        selection-background-color: #4299e1;
        color: #e8e8e8;
    }
    
    QHeaderView::section {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        color: white;
        padding: 8px;
        border: none;
        font-weight: 600;
    }
    
    QTableWidget::item {
        padding: 8px;
        border-bottom: 1px solid #4a5568;
    }
    
    QTableWidget::item:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        color: white;
    }
    
    /* Sliders */
    QSlider::groove:horizontal {
        height: 6px;
        background: #2d3748;
        border-radius: 3px;
        border: 1px solid #4a5568;
    }
    
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        border: 2px solid #1a202c;
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }
    
    QSlider::sub-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4299e1, stop:1 #63b3ed);
        border-radius: 3px;
    }
    
    /* Combo Boxes */
    QComboBox {
        background: #2d3748;
        border: 2px solid #4a5568;
        border-radius: 6px;
        padding: 6px 12px;
        color: #e8e8e8;
        min-width: 120px;
    }
    
    QComboBox:focus {
        border-color: #4299e1;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid #a0aec0;
        margin-right: 6px;
    }
    
    QComboBox QAbstractItemView {
        background: #2d3748;
        border: 1px solid #4299e1;
        border-radius: 6px;
        color: #e8e8e8;
        selection-background-color: #4299e1;
    }
    
    /* Spin Boxes */
    QSpinBox, QDoubleSpinBox {
        background: #2d3748;
        border: 2px solid #4a5568;
        border-radius: 6px;
        padding: 6px;
        color: #e8e8e8;
    }
    
    QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #4299e1;
    }
    
    /* Check Boxes */
    QCheckBox {
        color: #e8e8e8;
        font-weight: 500;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #4a5568;
        border-radius: 4px;
        background: #2d3748;
    }
    
    QCheckBox::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4299e1, stop:1 #3182ce);
        border-color: #4299e1;
    }
    
    QCheckBox::indicator:checked::after {
        content: "✓";
        color: white;
        font-weight: bold;
    }
    
    /* Labels */
    QLabel {
        color: #e8e8e8;
    }
    
    QLabel[class="title"] {
        color: #4299e1;
        font-size: 18px;
        font-weight: 700;
    }
    
    QLabel[class="subtitle"] {
        color: #a0aec0;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Splitter */
    QSplitter::handle {
        background: #4299e1;
        border-radius: 2px;
    }
    
    QSplitter::handle:horizontal {
        width: 4px;
    }
    
    QSplitter::handle:vertical {
        height: 4px;
    }
    
    /* Scroll Bars */
    QScrollBar:vertical {
        background: #2d3748;
        border: none;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background: #4299e1;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #63b3ed;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background: #2d3748;
        border: none;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background: #4299e1;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background: #63b3ed;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    """
    
    app.setStyleSheet(dark_stylesheet)

def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("PulseHunter")
    app.setApplicationVersion("Alpha-Enhanced-Functional")
    app.setOrganizationName("GeekAstro")
    app.setOrganizationDomain("geekastro.dev")

    # Set application style
    app.setStyle('Fusion')
    
    # Apply beautiful dark space theme
    apply_dark_theme(app)

    # Create and show main window
    try:
        window = PulseHunterMainWindow()
        window.show()

        # Run application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Critical error starting PulseHunter: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()