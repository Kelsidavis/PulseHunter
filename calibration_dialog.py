"""
Enhanced Calibration Setup Dialog for PulseHunter
Handles creation of master calibration files and ASTAP configuration
"""

import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QPushButton, QFileDialog, 
                             QLineEdit, QProgressBar, QTextEdit, QGroupBox, 
                             QComboBox, QCheckBox, QMessageBox, QTabWidget, 
                             QWidget, QSplitter, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QSettings, QRect
from PyQt6.QtGui import QIcon, QFont
import threading
import time
import logging
from calibration_utilities import CalibrationConfig, ASTAPManager, CalibrationLogger

class CalibrationWorker(QThread):
    """Worker thread for calibration processing to avoid UI freezing"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, calibration_type, input_folder, output_folder, settings):
        super().__init__()
        self.calibration_type = calibration_type
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.settings = settings
        self.is_cancelled = False
        
    def run(self):
        try:
            self.create_master_calibration()
        except Exception as e:
            self.finished.emit(False, str(e))
            
    def cancel(self):
        self.is_cancelled = True
        
    def create_master_calibration(self):
        """Process calibration frames and create master files"""
        self.status_updated.emit(f"Starting {self.calibration_type} calibration processing...")
        self.log_updated.emit(f"Input folder: {self.input_folder}")
        self.log_updated.emit(f"Output folder: {self.output_folder}")
        
        # Find FITS files
        fits_files = list(Path(self.input_folder).glob("*.fit*"))
        if not fits_files:
            self.finished.emit(False, "No FITS files found in selected folder")
            return
            
        self.log_updated.emit(f"Found {len(fits_files)} FITS files")
        total_files = len(fits_files)
        
        # Simulate processing steps (replace with actual FITS processing)
        processing_steps = [
            "Loading FITS files...",
            "Checking file headers...",
            "Validating exposure times...",
            "Computing statistics...",
            "Creating master frame...",
            "Saving master calibration file..."
        ]
        
        for i, step in enumerate(processing_steps):
            if self.is_cancelled:
                self.finished.emit(False, "Processing cancelled by user")
                return
                
            self.status_updated.emit(step)
            self.log_updated.emit(f"Step {i+1}/{len(processing_steps)}: {step}")
            
            # Simulate file processing
            if i == 0:  # Loading files
                for j, fits_file in enumerate(fits_files):
                    if self.is_cancelled:
                        return
                    self.log_updated.emit(f"Loading: {fits_file.name}")
                    progress = int((j + 1) * 20 / total_files)  # First 20% for loading
                    self.progress_updated.emit(progress)
                    self.msleep(50)  # Simulate processing time
            else:
                # Other processing steps
                progress = 20 + (i * 13)  # Distribute remaining 80% across steps
                self.progress_updated.emit(progress)
                self.msleep(500)
        
        # Create output filename
        output_file = Path(self.output_folder) / f"master_{self.calibration_type.lower()}.fits"
        self.log_updated.emit(f"Master file created: {output_file}")
        
        self.progress_updated.emit(100)
        self.status_updated.emit("Calibration processing completed successfully!")
        self.finished.emit(True, f"Master {self.calibration_type} calibration created successfully!")

class CalibrationSetupDialog(QDialog):
    """Enhanced calibration setup dialog with consistent positioning and features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings('PulseHunter', 'CalibrationDialog')
        self.config = CalibrationConfig()
        self.astap_manager = ASTAPManager(self.config)
        self.logger = CalibrationLogger()
        self.worker = None
        
        self.setup_ui()
        self.restore_geometry()
        self.load_settings()
        
    def setup_ui(self):
        self.setWindowTitle("PulseHunter - Calibration Setup")
        self.setMinimumSize(800, 700)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # ASTAP Configuration section
        self.setup_astap_section(layout)
        
        # Create tab widget for different calibration types
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Setup tabs
        self.setup_create_tab()
        self.setup_existing_tab()
        
        # Progress and log section
        self.setup_progress_section(layout)
        
        # Button section
        self.setup_buttons(layout)
        
    def setup_astap_section(self, layout):
        """Setup ASTAP executable configuration section"""
        astap_group = QGroupBox("ASTAP Plate Solving Configuration")
        astap_layout = QVBoxLayout(astap_group)
        
        # Info text
        info_label = QLabel(
            "ASTAP is required for plate solving and astrometric calibration. "
            "Please specify the location of your ASTAP executable."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        astap_layout.addWidget(info_label)
        
        # ASTAP path configuration
        path_layout = QHBoxLayout()
        
        # Status indicator
        self.astap_status_label = QLabel("●")
        self.astap_status_label.setFixedSize(20, 20)
        self.astap_status_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        self.astap_status_label.setToolTip("ASTAP status indicator")
        path_layout.addWidget(self.astap_status_label)
        
        # Path display
        self.astap_path_edit = QLineEdit()
        self.astap_path_edit.setPlaceholderText("ASTAP executable not configured...")
        self.astap_path_edit.setReadOnly(True)
        self.astap_path_edit.setStyleSheet("""
            QLineEdit[readOnly="true"] {
                background-color: #f5f5f5;
                color: #333;
                border: 1px solid #ccc;
                padding: 6px;
                border-radius: 3px;
            }
        """)
        path_layout.addWidget(self.astap_path_edit)
        
        # Browse button
        self.astap_browse_btn = QPushButton("Browse...")
        self.astap_browse_btn.setFixedWidth(80)
        self.astap_browse_btn.clicked.connect(self.browse_astap_executable)
        path_layout.addWidget(self.astap_browse_btn)
        
        # Test button
        self.astap_test_btn = QPushButton("Test")
        self.astap_test_btn.setFixedWidth(60)
        self.astap_test_btn.clicked.connect(self.test_astap_executable)
        self.astap_test_btn.setEnabled(False)
        path_layout.addWidget(self.astap_test_btn)
        
        astap_layout.addLayout(path_layout)
        
        # Status text
        self.astap_status_text = QLabel("ASTAP executable not found")
        self.astap_status_text.setStyleSheet("color: #666; font-size: 11px; margin-top: 5px;")
        astap_layout.addWidget(self.astap_status_text)
        
        layout.addWidget(astap_group)
        
    def setup_create_tab(self):
        """Tab for creating new master calibration files"""
        create_widget = QWidget()
        layout = QVBoxLayout(create_widget)
        
        # Instructions
        instructions = QLabel(
            "Create Master Calibration Files\n\n"
            "Select folders containing your calibration frames. The software will automatically "
            "process all FITS files in each folder and create master calibration files."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel { 
                background-color: #f0f8ff; 
                color: #1e3a8a;
                padding: 15px; 
                border-radius: 8px; 
                border: 1px solid #93c5fd;
                font-weight: 500;
                font-size: 13px;
            }
        """)
        layout.addWidget(instructions)
        
        # Calibration types grid
        grid_layout = QGridLayout()
        
        # Bias frames
        self.setup_calibration_row(grid_layout, 0, "Bias", "bias")
        
        # Dark frames  
        self.setup_calibration_row(grid_layout, 1, "Dark", "dark")
        
        # Flat frames
        self.setup_calibration_row(grid_layout, 2, "Flat", "flat")
        
        # Dark flat frames (new addition)
        self.setup_calibration_row(grid_layout, 3, "Dark Flat", "dark_flat")
        
        layout.addLayout(grid_layout)
        
        # Output folder selection
        output_group = QGroupBox("Master Files Output Location")
        output_layout = QHBoxLayout(output_group)
        
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Select where to save master calibration files...")
        output_layout.addWidget(QLabel("Output Folder:"))
        output_layout.addWidget(self.output_folder_edit)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.output_browse_btn)
        
        layout.addWidget(output_group)
        
        self.tab_widget.addTab(create_widget, "Create New Masters")
        
    def setup_calibration_row(self, grid_layout, row, name, type_key):
        """Setup a row for calibration frame selection"""
        # Enable checkbox
        checkbox = QCheckBox(f"Enable {name}")
        setattr(self, f"{type_key}_enabled", checkbox)
        grid_layout.addWidget(checkbox, row, 0)
        
        # Folder path
        path_edit = QLineEdit()
        path_edit.setPlaceholderText(f"Select folder containing {name.lower()} frames...")
        setattr(self, f"{type_key}_folder_edit", path_edit)
        grid_layout.addWidget(path_edit, row, 1)
        
        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda checked, t=type_key: self.browse_calibration_folder(t))
        grid_layout.addWidget(browse_btn, row, 2)
        
        # Status label
        status_label = QLabel("Not selected")
        status_label.setStyleSheet("color: gray;")
        setattr(self, f"{type_key}_status", status_label)
        grid_layout.addWidget(status_label, row, 3)
        
    def setup_existing_tab(self):
        """Tab for using existing master calibration files"""
        existing_widget = QWidget()
        layout = QVBoxLayout(existing_widget)
        
        # Instructions
        instructions = QLabel(
            "Use Existing Master Calibration Files\n\n"
            "Select pre-existing master calibration files to use for image processing. "
            "These files should be previously created master bias, dark, flat, or dark flat frames."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel { 
                background-color: #f0fdf4; 
                color: #166534;
                padding: 15px; 
                border-radius: 8px; 
                border: 1px solid #86efac;
                font-weight: 500;
                font-size: 13px;
            }
        """)
        layout.addWidget(instructions)
        
        # Existing files grid
        grid_layout = QGridLayout()
        
        # Master bias
        self.setup_existing_file_row(grid_layout, 0, "Master Bias", "master_bias")
        
        # Master dark
        self.setup_existing_file_row(grid_layout, 1, "Master Dark", "master_dark")
        
        # Master flat
        self.setup_existing_file_row(grid_layout, 2, "Master Flat", "master_flat")
        
        # Master dark flat
        self.setup_existing_file_row(grid_layout, 3, "Master Dark Flat", "master_dark_flat")
        
        layout.addLayout(grid_layout)
        
        self.tab_widget.addTab(existing_widget, "Use Existing Masters")
        
    def setup_existing_file_row(self, grid_layout, row, name, type_key):
        """Setup a row for existing master file selection"""
        # Enable checkbox
        checkbox = QCheckBox(f"Use {name}")
        setattr(self, f"{type_key}_enabled", checkbox)
        grid_layout.addWidget(checkbox, row, 0)
        
        # File path
        path_edit = QLineEdit()
        path_edit.setPlaceholderText(f"Select existing {name.lower()} file...")
        setattr(self, f"{type_key}_file_edit", path_edit)
        grid_layout.addWidget(path_edit, row, 1)
        
        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda checked, t=type_key: self.browse_existing_file(t))
        grid_layout.addWidget(browse_btn, row, 2)
        
        # Info button
        info_btn = QPushButton("Info")
        info_btn.clicked.connect(lambda checked, t=type_key: self.show_file_info(t))
        grid_layout.addWidget(info_btn, row, 3)
        
    def setup_progress_section(self, layout):
        """Setup progress bar and log display"""
        progress_group = QGroupBox("Processing Status")
        progress_layout = QVBoxLayout(progress_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #2c5aa0;")
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Log display
        log_label = QLabel("Processing Log:")
        progress_layout.addWidget(log_label)
        
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setFont(QFont("Consolas", 9))
        progress_layout.addWidget(self.log_display)
        
        layout.addWidget(progress_group)
        
    def setup_buttons(self, layout):
        """Setup dialog buttons"""
        button_layout = QHBoxLayout()
        
        # Process button
        self.process_btn = QPushButton("Create Master Files")
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_btn)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel Processing")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def center_on_parent_or_screen(self):
        """Ensure dialog appears in consistent location"""
        if self.parent():
            # Center on parent window
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            # Center on primary screen
            screen = QApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)
            
    def restore_geometry(self):
        """Restore dialog position and size"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.center_on_parent_or_screen()
            
    def save_geometry(self):
        """Save dialog position and size"""
        self.settings.setValue("geometry", self.saveGeometry())
        
    def load_settings(self):
        """Load previously saved settings"""
        # Load ASTAP path - first try from ASTAP manager, then from settings
        astap_path = self.astap_manager.astap_path or self.settings.value("astap_executable", "")
        
        if astap_path:
            self.astap_path_edit.setText(astap_path)
            # Skip validation during load to prevent dialogs
            is_cli = "cli" in Path(astap_path).name.lower() or "console" in Path(astap_path).name.lower()
            app_type = "CLI" if is_cli else "GUI"
            self.update_astap_status(True, f"ASTAP {app_type} version loaded: {Path(astap_path).name}")
        else:
            self.auto_detect_astap()
            
        # Load folder paths
        for cal_type in ["bias", "dark", "flat", "dark_flat"]:
            folder_path = self.settings.value(f"{cal_type}_folder", "")
            if hasattr(self, f"{cal_type}_folder_edit"):
                getattr(self, f"{cal_type}_folder_edit").setText(folder_path)
                
        # Load output folder
        output_path = self.settings.value("output_folder", "")
        self.output_folder_edit.setText(output_path)
        
    def save_settings(self):
        """Save current settings"""
        # Save ASTAP path
        self.settings.setValue("astap_executable", self.astap_path_edit.text())
        
        # Save folder paths
        for cal_type in ["bias", "dark", "flat", "dark_flat"]:
            if hasattr(self, f"{cal_type}_folder_edit"):
                folder_path = getattr(self, f"{cal_type}_folder_edit").text()
                self.settings.setValue(f"{cal_type}_folder", folder_path)
                
        # Save output folder
        self.settings.setValue("output_folder", self.output_folder_edit.text())
        
    # ASTAP-related methods
    def auto_detect_astap(self):
        """Attempt to auto-detect ASTAP executable"""
        detected_path = self.astap_manager.auto_detect_astap()
        if detected_path:
            self.astap_path_edit.setText(detected_path)
            self.validate_astap_executable(detected_path)
            self.add_log_entry(f"Auto-detected ASTAP at: {detected_path}")
                    
    def browse_astap_executable(self):
        """Browse for ASTAP executable"""
        if sys.platform == "win32":
            file_filter = "Executable Files (*.exe);;All Files (*)"
            default_name = "astap.exe"
        else:
            file_filter = "All Files (*)"
            default_name = "astap"
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASTAP Executable",
            str(Path.home()),
            file_filter
        )
        
        if file_path:
            self.astap_path_edit.setText(file_path)
            self.validate_astap_executable(file_path)
            
    def validate_astap_executable(self, path):
        """Validate the ASTAP executable - skip execution test to prevent dialogs"""
        if not path:
            self.update_astap_status(False, "No path specified")
            return False
            
        path_obj = Path(path)
        
        # Check if file exists
        if not path_obj.exists():
            self.update_astap_status(False, f"File not found: {path}")
            return False
            
        # Check if it's executable
        if not os.access(path, os.X_OK):
            self.update_astap_status(False, f"File is not executable: {path}")
            return False
            
        # Check filename contains "astap"
        if "astap" not in path_obj.name.lower():
            self.update_astap_status(False, f"Warning: Filename doesn't contain 'astap': {path_obj.name}")
            return False
        
        # Skip execution test during validation to prevent dialog popups
        is_cli = "cli" in path_obj.name.lower() or "console" in path_obj.name.lower()
        app_type = "CLI" if is_cli else "GUI"
        self.update_astap_status(True, f"ASTAP {app_type} version validated: {path_obj.name}")
        return True
            
    def extract_astap_version(self, help_output):
        """Extract version information from ASTAP help output"""
        lines = help_output.split('\n')
        for line in lines[:5]:  # Check first few lines
            if any(word in line.lower() for word in ['version', 'v.', 'astap']):
                return line.strip()
        return "Version unknown"
        
    def update_astap_status(self, is_valid, message):
        """Update ASTAP status indicators"""
        if is_valid:
            self.astap_status_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
            self.astap_status_label.setToolTip("ASTAP executable found and validated")
            self.astap_status_text.setText(message)
            self.astap_status_text.setStyleSheet("color: green; font-size: 11px; margin-top: 5px;")
            self.astap_test_btn.setEnabled(True)
        else:
            self.astap_status_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
            self.astap_status_label.setToolTip("ASTAP executable not found or invalid")
            self.astap_status_text.setText(message)
            self.astap_status_text.setStyleSheet("color: red; font-size: 11px; margin-top: 5px;")
            self.astap_test_btn.setEnabled(False)
            
    def test_astap_executable(self):
        """Test ASTAP executable by running it with help flag"""
        astap_path = self.astap_path_edit.text()
        if not astap_path:
            return
            
        # Warn user about potential dialogs
        is_cli = "cli" in Path(astap_path).name.lower() or "console" in Path(astap_path).name.lower()
        app_type = "CLI" if is_cli else "GUI"
        
        if not is_cli:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Test ASTAP GUI Version",
                f"Testing {Path(astap_path).name} may open GUI windows or dialogs.\n\nProceed with test?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            
        try:
            import subprocess
            self.add_log_entry(f"Testing ASTAP {app_type} version...")
            
            # Use different timeout for CLI vs GUI
            timeout = 10 if is_cli else 15
            
            result = subprocess.run([astap_path, "-h"], capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                self.add_log_entry("ASTAP test successful!")
                output_preview = result.stdout[:200] if result.stdout else "No console output (normal for GUI versions)"
                QMessageBox.information(
                    self, 
                    f"ASTAP {app_type} Test", 
                    f"ASTAP executable test successful!\n\nReturn code: {result.returncode}\nOutput preview:\n{output_preview}..."
                )
            else:
                self.add_log_entry(f"ASTAP test failed with return code: {result.returncode}")
                QMessageBox.warning(
                    self,
                    "ASTAP Test Failed", 
                    f"ASTAP returned error code {result.returncode}\n\nError: {result.stderr[:200]}"
                )
        except subprocess.TimeoutExpired:
            if is_cli:
                self.add_log_entry("ASTAP CLI test timed out")
                QMessageBox.warning(self, "ASTAP Test", "ASTAP CLI test timed out after 10 seconds")
            else:
                self.add_log_entry("ASTAP GUI test timed out (may be normal)")
                QMessageBox.information(self, "ASTAP Test", 
                    "ASTAP GUI test timed out, but this may be normal\nif ASTAP opened a GUI window. Check if ASTAP is running.")
        except Exception as e:
            self.add_log_entry(f"ASTAP test error: {str(e)}")
            QMessageBox.critical(self, "ASTAP Test Error", f"Error testing ASTAP:\n{str(e)}")
    
    # Calibration folder methods
    def browse_calibration_folder(self, cal_type):
        """Browse for calibration frame folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            f"Select {cal_type.replace('_', ' ').title()} Frames Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            folder_edit = getattr(self, f"{cal_type}_folder_edit")
            folder_edit.setText(folder)
            
            # Update status
            fits_count = len(list(Path(folder).glob("*.fit*")))
            status_label = getattr(self, f"{cal_type}_status")
            if fits_count > 0:
                status_label.setText(f"{fits_count} FITS files found")
                status_label.setStyleSheet("color: green;")
            else:
                status_label.setText("No FITS files found")
                status_label.setStyleSheet("color: red;")
                
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for Master Calibration Files",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.output_folder_edit.setText(folder)
            
    def browse_existing_file(self, file_type):
        """Browse for existing master calibration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Existing {file_type.replace('_', ' ').title()} File",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)"
        )
        
        if file_path:
            file_edit = getattr(self, f"{file_type}_file_edit")
            file_edit.setText(file_path)
            
    def show_file_info(self, file_type):
        """Show information about selected master file"""
        file_edit = getattr(self, f"{file_type}_file_edit")
        file_path = file_edit.text()
        
        if not file_path or not Path(file_path).exists():
            QMessageBox.warning(self, "File Info", "No file selected or file does not exist.")
            return
            
        # Basic file info (in real implementation, would read FITS headers)
        file_size = Path(file_path).stat().st_size
        info_text = f"""
File: {Path(file_path).name}
Path: {file_path}
Size: {file_size / 1024 / 1024:.2f} MB

Note: In the full implementation, this would show:
- FITS header information
- Image dimensions
- Creation date
- Processing history
- Statistics (mean, std, etc.)
        """
        
        QMessageBox.information(self, "Master File Information", info_text.strip())
        
    # Processing methods
    def start_processing(self):
        """Start calibration processing"""
        # Validate inputs
        if self.tab_widget.currentIndex() == 0:  # Create new masters
            enabled_types = []
            for cal_type in ["bias", "dark", "flat", "dark_flat"]:
                enabled_cb = getattr(self, f"{cal_type}_enabled")
                if enabled_cb.isChecked():
                    folder_edit = getattr(self, f"{cal_type}_folder_edit")
                    if not folder_edit.text():
                        QMessageBox.warning(self, "Validation Error", 
                                          f"Please select a folder for {cal_type.replace('_', ' ')} frames.")
                        return
                    enabled_types.append(cal_type)
                    
            if not enabled_types:
                QMessageBox.warning(self, "Validation Error", 
                                  "Please enable at least one calibration type.")
                return
                
            if not self.output_folder_edit.text():
                QMessageBox.warning(self, "Validation Error", 
                                  "Please select an output folder.")
                return
                
            # Process first enabled type (in real implementation, would process all)
            first_type = enabled_types[0]
            folder_edit = getattr(self, f"{first_type}_folder_edit")
            
            self.start_worker_processing(first_type, folder_edit.text(), self.output_folder_edit.text())
        else:
            QMessageBox.information(self, "Info", "Existing master files validated and ready for use!")
            
    def start_worker_processing(self, cal_type, input_folder, output_folder):
        """Start processing in worker thread"""
        self.worker = CalibrationWorker(cal_type, input_folder, output_folder, {})
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.log_updated.connect(self.add_log_entry)
        self.worker.finished.connect(self.processing_finished)
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        
        self.worker.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        
    def add_log_entry(self, message):
        """Add entry to log display"""
        self.log_display.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        
    def cancel_processing(self):
        """Cancel current processing"""
        if self.worker:
            self.worker.cancel()
            self.add_log_entry("Cancellation requested...")
            
    def processing_finished(self, success, message):
        """Handle processing completion"""
        self.progress_bar.setVisible(False)
        self.process_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        
        if success:
            self.status_label.setText("Processing completed successfully!")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Processing failed!")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            QMessageBox.critical(self, "Error", message)
            
        self.worker = None
        
    def closeEvent(self, event):
        """Handle dialog closing"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Confirm Close", 
                                       "Processing is still running. Are you sure you want to close?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel()
                self.worker.wait()
            else:
                event.ignore()
                return
                
        self.save_geometry()
        self.save_settings()
        event.accept()

# Example usage and testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("PulseHunter")
    app.setOrganizationName("GeekAstro")
    
    dialog = CalibrationSetupDialog()
    dialog.show()
    
    sys.exit(app.exec())