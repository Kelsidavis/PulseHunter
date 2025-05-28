"""
Fixed PulseHunter Calibration Dialog
Now properly handles lights folder selection and automatic master file usage
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QRect, QSettings, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calibration_utilities import ASTAPManager, CalibrationConfig, CalibrationLogger
from fits_processing import CalibrationProcessor


class CalibrationProject:
    """Manages calibration project data and automatic master file usage"""

    def __init__(self):
        self.config_file = Path("calibration_projects.json")
        self.projects = self.load_projects()

    def load_projects(self):
        """Load existing calibration projects"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_projects(self):
        """Save calibration projects"""
        with open(self.config_file, "w") as f:
            json.dump(self.projects, f, indent=2)

    def create_project(self, lights_folder, master_files):
        """Create a new calibration project"""
        project_id = str(Path(lights_folder).resolve())
        self.projects[project_id] = {
            "lights_folder": lights_folder,
            "master_files": master_files,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        self.save_projects()
        return project_id

    def get_masters_for_folder(self, lights_folder):
        """Get master files for a specific lights folder"""
        project_id = str(Path(lights_folder).resolve())
        if project_id in self.projects:
            project = self.projects[project_id]
            project["last_used"] = datetime.now().isoformat()
            self.save_projects()
            return project.get("master_files", {})
        return {}


class CalibrationWorker(QThread):
    """Worker thread for calibration processing"""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)  # success, message, master_files

    def __init__(self, calibration_tasks, output_folder):
        super().__init__()
        self.calibration_tasks = calibration_tasks
        self.output_folder = Path(output_folder)
        self.is_cancelled = False
        self.master_files = {}

    def run(self):
        """Main processing function"""
        try:
            self.status_updated.emit("Starting calibration processing...")
            self.log_updated.emit("Beginning calibration file creation...")

            total_tasks = len(
                [task for task in self.calibration_tasks if task["enabled"]]
            )
            completed_tasks = 0

            for task in self.calibration_tasks:
                if not task["enabled"] or self.is_cancelled:
                    continue

                cal_type = task["type"]
                input_folder = task["folder"]

                self.status_updated.emit(f"Processing {cal_type} frames...")
                self.log_updated.emit(f"Creating master {cal_type} from {input_folder}")

                # Create master file
                output_file = self.output_folder / f"master_{cal_type}.fits"

                processor = CalibrationProcessor()
                input_files = list(Path(input_folder).glob("*.fit*"))

                if not input_files:
                    self.log_updated.emit(f"No FITS files found in {input_folder}")
                    continue

                def progress_callback(value):
                    overall_progress = int(
                        (completed_tasks / total_tasks) * 100 + (value / total_tasks)
                    )
                    self.progress_updated.emit(overall_progress)

                success = processor.create_master_calibration(
                    input_files, output_file, cal_type, progress_callback
                )

                if success:
                    self.master_files[cal_type] = str(output_file)
                    self.log_updated.emit(
                        f"✅ Master {cal_type} created: {output_file.name}"
                    )
                else:
                    self.log_updated.emit(f"❌ Failed to create master {cal_type}")

                completed_tasks += 1

            self.progress_updated.emit(100)
            self.status_updated.emit("Calibration processing completed!")

            message = f"Created {len(self.master_files)} master calibration files"
            self.finished.emit(True, message, self.master_files)

        except Exception as e:
            error_msg = f"Calibration processing error: {str(e)}"
            self.log_updated.emit(f"ERROR: {error_msg}")
            self.finished.emit(False, error_msg, {})

    def cancel(self):
        self.is_cancelled = True


class FixedCalibrationDialog(QDialog):
    """Fixed calibration dialog with lights folder selection and automatic master usage"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("PulseHunter", "FixedCalibrationDialog")
        self.config = CalibrationConfig()
        self.astap_manager = ASTAPManager(self.config)
        self.logger = CalibrationLogger()
        self.worker = None
        self.calibration_project = CalibrationProject()

        self.setup_ui()
        self.restore_geometry()
        self.load_settings()

    def setup_ui(self):
        self.setWindowTitle("PulseHunter - Fixed Calibration Setup")
        self.setMinimumSize(900, 800)

        # Main layout
        layout = QVBoxLayout(self)

        # Step 1: Lights Folder Selection (NEW - This is the key fix!)
        self.setup_lights_folder_section(layout)

        # Step 2: ASTAP Configuration
        self.setup_astap_section(layout)

        # Step 3: Calibration Frames
        self.setup_calibration_frames_section(layout)

        # Step 4: Processing
        self.setup_processing_section(layout)

        # Buttons
        self.setup_buttons(layout)

    def setup_lights_folder_section(self, layout):
        """NEW: Setup lights folder selection - this is the main fix"""
        lights_group = QGroupBox("🔬 Step 1: Select Your Science Images (Lights)")
        lights_layout = QVBoxLayout(lights_group)

        # Instructions
        instructions = QLabel(
            "Select the folder containing your science images (lights) that need to be calibrated.\n"
            "The master calibration files will be automatically applied to these images."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            """
            QLabel {
                background-color: #e8f4fd;
                color: #1565c0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #42a5f5;
                font-weight: 500;
                margin-bottom: 10px;
            }
            """
        )
        lights_layout.addWidget(instructions)

        # Lights folder selection
        folder_layout = QHBoxLayout()

        folder_layout.addWidget(QLabel("Lights Folder:"))

        self.lights_folder_edit = QLineEdit()
        self.lights_folder_edit.setPlaceholderText(
            "Select folder containing your science images..."
        )
        self.lights_folder_edit.textChanged.connect(self.check_existing_calibration)
        folder_layout.addWidget(self.lights_folder_edit)

        browse_lights_btn = QPushButton("Browse...")
        browse_lights_btn.clicked.connect(self.browse_lights_folder)
        folder_layout.addWidget(browse_lights_btn)

        lights_layout.addLayout(folder_layout)

        # Existing calibration status
        self.existing_cal_label = QLabel(
            "Select a lights folder to check for existing calibration"
        )
        self.existing_cal_label.setStyleSheet(
            "color: #666; font-style: italic; padding: 5px;"
        )
        lights_layout.addWidget(self.existing_cal_label)

        layout.addWidget(lights_group)

    def setup_astap_section(self, layout):
        """Setup ASTAP configuration"""
        astap_group = QGroupBox("⭐ Step 2: ASTAP Plate Solving (Optional)")
        astap_layout = QVBoxLayout(astap_group)

        # ASTAP path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("ASTAP Path:"))

        self.astap_path_edit = QLineEdit()
        self.astap_path_edit.setPlaceholderText("ASTAP executable path (auto-detected)")
        self.astap_path_edit.setReadOnly(True)
        path_layout.addWidget(self.astap_path_edit)

        astap_browse_btn = QPushButton("Browse...")
        astap_browse_btn.clicked.connect(self.browse_astap)
        path_layout.addWidget(astap_browse_btn)

        auto_detect_btn = QPushButton("Auto-Detect")
        auto_detect_btn.clicked.connect(self.auto_detect_astap)
        path_layout.addWidget(auto_detect_btn)

        astap_layout.addLayout(path_layout)

        # Status
        self.astap_status_label = QLabel("ASTAP not configured")
        self.astap_status_label.setStyleSheet("color: #666;")
        astap_layout.addWidget(self.astap_status_label)

        layout.addWidget(astap_group)

    def setup_calibration_frames_section(self, layout):
        """Setup calibration frames selection"""
        cal_group = QGroupBox("📸 Step 3: Select Calibration Frames")
        cal_layout = QVBoxLayout(cal_group)

        # Instructions
        instructions = QLabel(
            "Select folders containing your calibration frames. Enable only the types you have available."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin-bottom: 10px;")
        cal_layout.addWidget(instructions)

        # Grid for calibration types
        grid_layout = QGridLayout()

        # Setup each calibration type
        self.calibration_controls = {}
        cal_types = [
            ("bias", "Bias Frames", "Dark current and read noise calibration"),
            (
                "dark",
                "Dark Frames",
                "Thermal noise calibration (same exposure as lights)",
            ),
            ("flat", "Flat Frames", "Vignetting and dust spot correction"),
            ("dark_flat", "Dark Flat Frames", "Dark correction for flat frames"),
        ]

        for i, (cal_type, name, description) in enumerate(cal_types):
            # Enable checkbox
            checkbox = QCheckBox(f"Use {name}")
            checkbox.setToolTip(description)
            grid_layout.addWidget(checkbox, i, 0)

            # Folder selection
            folder_edit = QLineEdit()
            folder_edit.setPlaceholderText(f"Select {name.lower()} folder...")
            grid_layout.addWidget(folder_edit, i, 1)

            # Browse button
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda checked, t=cal_type: self.browse_calibration_folder(t)
            )
            grid_layout.addWidget(browse_btn, i, 2)

            # Status
            status_label = QLabel("Not selected")
            status_label.setStyleSheet("color: #666;")
            grid_layout.addWidget(status_label, i, 3)

            # Store controls
            self.calibration_controls[cal_type] = {
                "checkbox": checkbox,
                "folder_edit": folder_edit,
                "browse_btn": browse_btn,
                "status_label": status_label,
            }

        cal_layout.addLayout(grid_layout)

        # Output folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Master Files Output:"))

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText(
            "Where to save master calibration files..."
        )
        output_layout.addWidget(self.output_folder_edit)

        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(output_browse_btn)

        cal_layout.addLayout(output_layout)

        layout.addWidget(cal_group)

    def setup_processing_section(self, layout):
        """Setup processing section"""
        processing_group = QGroupBox("⚙️ Step 4: Processing Status")
        processing_layout = QVBoxLayout(processing_group)

        # Status
        self.status_label = QLabel("Ready to create master calibration files")
        self.status_label.setStyleSheet("font-weight: bold; color: #2c5aa0;")
        processing_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        processing_layout.addWidget(self.progress_bar)

        # Log
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setReadOnly(True)
        processing_layout.addWidget(self.log_display)

        layout.addWidget(processing_group)

    def setup_buttons(self, layout):
        """Setup dialog buttons"""
        button_layout = QHBoxLayout()

        # Create masters button
        self.create_btn = QPushButton("🚀 Create Master Files & Setup Project")
        self.create_btn.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                padding: 12px 24px;
                font-weight: bold;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }
            QPushButton:disabled {
                background: #cccccc;
                color: #666;
            }
            """
        )
        self.create_btn.clicked.connect(self.create_master_files)
        button_layout.addWidget(self.create_btn)

        # Cancel button
        self.cancel_btn = QPushButton("⏹️ Cancel Processing")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        button_layout.addWidget(self.cancel_btn)

        button_layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    # Event handlers
    def browse_lights_folder(self):
        """Browse for lights folder - KEY NEW FUNCTIONALITY"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Lights Folder (Science Images)", ""
        )
        if folder:
            self.lights_folder_edit.setText(folder)

            # Count FITS files
            fits_count = len(list(Path(folder).glob("*.fit*")))
            self.add_log(
                f"Selected lights folder: {Path(folder).name} ({fits_count} FITS files)"
            )

            # Set default output folder
            if not self.output_folder_edit.text():
                default_output = Path(folder).parent / "master_calibrations"
                self.output_folder_edit.setText(str(default_output))

    def check_existing_calibration(self):
        """Check if calibration already exists for this lights folder"""
        lights_folder = self.lights_folder_edit.text()
        if not lights_folder:
            self.existing_cal_label.setText(
                "Select a lights folder to check for existing calibration"
            )
            self.existing_cal_label.setStyleSheet("color: #666; font-style: italic;")
            return

        existing_masters = self.calibration_project.get_masters_for_folder(
            lights_folder
        )

        if existing_masters:
            # Found existing calibration
            master_list = []
            for cal_type, file_path in existing_masters.items():
                if Path(file_path).exists():
                    master_list.append(f"{cal_type}: ✅")
                else:
                    master_list.append(f"{cal_type}: ❌ (missing)")

            self.existing_cal_label.setText(
                f"🎯 Existing calibration found! Masters: {', '.join(master_list)}"
            )
            self.existing_cal_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.existing_cal_label.setText(
                "No existing calibration found - will create new master files"
            )
            self.existing_cal_label.setStyleSheet("color: #666; font-style: italic;")

    def browse_astap(self):
        """Browse for ASTAP executable"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASTAP Executable",
            "",
            "Executable Files (*.exe);;All Files (*)"
            if sys.platform == "win32"
            else "All Files (*)",
        )
        if file_path:
            self.astap_path_edit.setText(file_path)
            self.validate_astap()

    def auto_detect_astap(self):
        """Auto-detect ASTAP"""
        self.add_log("Auto-detecting ASTAP...")
        detected_path = self.astap_manager.auto_detect_astap()
        if detected_path:
            self.astap_path_edit.setText(detected_path)
            self.validate_astap()
            self.add_log(f"ASTAP auto-detected: {Path(detected_path).name}")
        else:
            self.add_log("ASTAP not found - plate solving will be unavailable")

    def validate_astap(self):
        """Validate ASTAP configuration"""
        astap_path = self.astap_path_edit.text()
        if astap_path and Path(astap_path).exists():
            self.astap_status_label.setText("✅ ASTAP configured")
            self.astap_status_label.setStyleSheet("color: green;")
        else:
            self.astap_status_label.setText("❌ ASTAP not found")
            self.astap_status_label.setStyleSheet("color: red;")

    def browse_calibration_folder(self, cal_type):
        """Browse for calibration folder"""
        controls = self.calibration_controls[cal_type]

        folder = QFileDialog.getExistingDirectory(
            self, f"Select {cal_type.title()} Frames Folder", ""
        )
        if folder:
            controls["folder_edit"].setText(folder)

            # Count FITS files
            fits_count = len(list(Path(folder).glob("*.fit*")))
            if fits_count > 0:
                controls["status_label"].setText(f"✅ {fits_count} files")
                controls["status_label"].setStyleSheet("color: green;")
                controls["checkbox"].setChecked(True)
            else:
                controls["status_label"].setText("❌ No FITS files")
                controls["status_label"].setStyleSheet("color: red;")

    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Master Files", ""
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def create_master_files(self):
        """Create master calibration files and setup project"""
        # Validate inputs
        if not self.lights_folder_edit.text():
            QMessageBox.warning(
                self, "Missing Input", "Please select your lights folder first!"
            )
            return

        if not self.output_folder_edit.text():
            QMessageBox.warning(
                self,
                "Missing Output",
                "Please select an output folder for master files!",
            )
            return

        # Check which calibration types are enabled
        calibration_tasks = []
        for cal_type, controls in self.calibration_controls.items():
            if controls["checkbox"].isChecked():
                folder = controls["folder_edit"].text()
                if not folder:
                    QMessageBox.warning(
                        self,
                        "Missing Folder",
                        f"Please select a folder for {cal_type} frames or disable this option!",
                    )
                    return

                calibration_tasks.append(
                    {"type": cal_type, "folder": folder, "enabled": True}
                )

        if not calibration_tasks:
            QMessageBox.warning(
                self,
                "No Calibration Selected",
                "Please enable at least one type of calibration frame!",
            )
            return

        # Start processing
        self.start_processing(calibration_tasks)

    def start_processing(self, calibration_tasks):
        """Start the calibration processing"""
        output_folder = self.output_folder_edit.text()
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Start worker thread
        self.worker = CalibrationWorker(calibration_tasks, output_folder)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.log_updated.connect(self.add_log)
        self.worker.finished.connect(self.processing_finished)

        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.create_btn.setVisible(False)
        self.cancel_btn.setVisible(True)

        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.add_log("Processing cancelled by user")

    def processing_finished(self, success, message, master_files):
        """Handle processing completion - KEY NEW FUNCTIONALITY"""
        # Update UI
        self.progress_bar.setVisible(False)
        self.create_btn.setVisible(True)
        self.cancel_btn.setVisible(False)

        if success and master_files:
            self.status_label.setText("✅ Master files created successfully!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            # Create calibration project - THIS IS THE KEY FIX!
            lights_folder = self.lights_folder_edit.text()
            project_id = self.calibration_project.create_project(
                lights_folder, master_files
            )

            self.add_log(
                f"✅ Calibration project created for: {Path(lights_folder).name}"
            )
            self.add_log(
                f"Master files will be automatically used when processing these lights!"
            )

            # Show success message
            master_list = "\n".join(
                [
                    f"• {cal_type}: {Path(path).name}"
                    for cal_type, path in master_files.items()
                ]
            )

            QMessageBox.information(
                self,
                "Calibration Complete!",
                f"🎉 Master calibration files created successfully!\n\n"
                f"Created files:\n{master_list}\n\n"
                f"✨ These will be automatically applied when you process the images in:\n"
                f"{Path(lights_folder).name}\n\n"
                f"You can now close this dialog and process your images!",
            )
        else:
            self.status_label.setText("❌ Processing failed!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Processing Failed", f"Error: {message}")

        # Cleanup
        self.worker = None

    def load_settings(self):
        """Load previous settings"""
        # Auto-detect ASTAP on startup
        astap_path = self.astap_manager.astap_path or self.settings.value(
            "astap_path", ""
        )
        if astap_path:
            self.astap_path_edit.setText(astap_path)
            self.validate_astap()
        else:
            self.auto_detect_astap()

        # Load folder paths
        for cal_type in self.calibration_controls:
            folder_path = self.settings.value(f"{cal_type}_folder", "")
            if folder_path:
                self.calibration_controls[cal_type]["folder_edit"].setText(folder_path)

        # Load output folder
        output_path = self.settings.value("output_folder", "")
        if output_path:
            self.output_folder_edit.setText(output_path)

    def save_settings(self):
        """Save current settings"""
        self.settings.setValue("astap_path", self.astap_path_edit.text())

        for cal_type in self.calibration_controls:
            folder_path = self.calibration_controls[cal_type]["folder_edit"].text()
            self.settings.setValue(f"{cal_type}_folder", folder_path)

        self.settings.setValue("output_folder", self.output_folder_edit.text())

    def restore_geometry(self):
        """Restore dialog geometry"""
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
        """Handle dialog closing"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing Active",
                "Calibration processing is still running. Cancel and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel()
                self.worker.wait()
            else:
                event.ignore()
                return

        self.settings.setValue("geometry", self.saveGeometry())
        self.save_settings()
        event.accept()


# Integration function to get master files for processing
def get_master_files_for_folder(lights_folder):
    """
    Get master calibration files for a specific lights folder
    This function should be called from the main processing code
    """
    project = CalibrationProject()
    return project.get_masters_for_folder(lights_folder)


# Test the dialog
if __name__ == "__main__":
    app = QApplication(sys.argv)

    dialog = FixedCalibrationDialog()
    dialog.show()

    sys.exit(app.exec())
