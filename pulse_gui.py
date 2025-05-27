"""
PulseHunter Main GUI Application
Enhanced with improved calibration dialog and ASTAP integration
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMenuBar, QVBoxLayout, QWidget, QStatusBar, QMessageBox, QLabel, QHBoxLayout, QGroupBox, QPushButton, QTextEdit, QSplitter, QTabWidget, QProgressBar, QFileDialog, QDialog)
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSettings, QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QPixmap

# Import enhanced calibration components
from calibration_dialog import CalibrationSetupDialog
from calibration_utilities import (CalibrationConfig, ASTAPManager,
                                   CalibrationLogger, DialogPositionManager)

class PulseHunterMainWindow(QMainWindow):
    """Enhanced PulseHunter main window with ASTAP integration"""

    def __init__(self):
        super().__init__()

        # Initialize core components
        self.settings = QSettings('PulseHunter', 'MainApplication')
        self.config = CalibrationConfig()
        self.astap_manager = ASTAPManager(self.config)
        self.logger = CalibrationLogger()

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
        self.setWindowTitle("PulseHunter - Optical SETI & Exoplanet Detection")
        self.setMinimumSize(1200, 800)

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

        # System status
        self.system_status_label = QLabel("System: Ready")
        self.system_status_label.setStyleSheet("color: #666; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.system_status_label)

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
        welcome_group = QGroupBox("Welcome to PulseHunter")
        welcome_layout = QVBoxLayout(welcome_group)

        welcome_text = QLabel("""
        <h2>PulseHunter - Optical SETI & Exoplanet Detection Pipeline</h2>
        <p>Welcome to PulseHunter, your gateway to citizen science astronomy!</p>

        <h3>Getting Started:</h3>
        <ol>
        <li><b>Configure ASTAP:</b> Set up plate solving (Calibration → Configure ASTAP)</li>
        <li><b>Setup Calibration:</b> Create master calibration files (Calibration → Calibration Setup)</li>
        <li><b>Process Images:</b> Analyze your FITS files for detections (Processing → Process Images)</li>
        <li><b>Review Results:</b> Examine potential discoveries (Analysis → View Results)</li>
        </ol>

        <p><i>Join the global network of citizen scientists searching for extraterrestrial intelligence
        and discovering new exoplanets!</i></p>
        """)
        welcome_text.setWordWrap(True)
        welcome_layout.addWidget(welcome_text)

        layout.addWidget(welcome_group)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)

        calibration_btn = QPushButton("Setup Calibration")
        calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        calibration_btn.clicked.connect(self.open_calibration_dialog)
        actions_layout.addWidget(calibration_btn)

        process_btn = QPushButton("Process Images")
        process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        process_btn.clicked.connect(self.process_images)
        actions_layout.addWidget(process_btn)

        results_btn = QPushButton("View Results")
        results_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        results_btn.clicked.connect(self.view_results)
        actions_layout.addWidget(results_btn)

        actions_layout.addStretch()
        layout.addWidget(actions_group)

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
        """Setup results tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)

        # Results placeholder
        results_label = QLabel("""
        <h3>Detection Results</h3>
        <p>Results from processed images will appear here.</p>
        <p>Use Processing → Process Images to analyze your FITS files.</p>
        """)
        results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_label.setStyleSheet("color: #666; padding: 50px;")
        layout.addWidget(results_label)

        self.tab_widget.addTab(results_widget, "Results")

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

    def add_system_log(self, message):
        """Add message to system log"""
        timestamp = QTimer().singleShot(0, lambda: None)  # Get current time
        import datetime
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.system_log.append(f"[{time_str}] {message}")

    # Menu action handlers
    def new_project(self):
        """Create new project"""
        QMessageBox.information(self, "New Project", "New project functionality will be implemented here.")

    def open_project(self):
        """Open existing project"""
        QMessageBox.information(self, "Open Project", "Open project functionality will be implemented here.")

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
        """Process FITS images"""
        self.add_system_log("Starting image processing...")
        QMessageBox.information(
            self,
            "Process Images",
            "Image processing functionality will be implemented here.\n\n"
            "This will analyze FITS files for optical transients and exoplanet transits."
        )

    def view_results(self):
        """View analysis results"""
        self.add_system_log("Opening results viewer...")
        QMessageBox.information(
            self,
            "View Results",
            "Results viewer will be implemented here.\n\n"
            "This will show detected events, statistics, and export options."
        )

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
        from PyQt6.QtGui import QAction
        from PyQt6.QtCore import QUrl

        # Try to open online documentation
        QDesktopServices.openUrl(QUrl("https://github.com/Kelsidavis/PulseHunter"))

    def show_about(self):
        """Show about dialog"""
        astap_status = "✓ Configured" if self.astap_manager.is_configured() else "✗ Not configured"

        QMessageBox.about(
            self,
            "About PulseHunter",
            f"""
            <h3>PulseHunter</h3>
            <p><b>Optical SETI and Exoplanet Transit Detection Pipeline</b></p>
            <p>Version: Alpha (Enhanced Calibration)</p>

            <p>PulseHunter empowers amateur astronomers worldwide to contribute
            to cutting-edge astronomical research through citizen science.</p>

            <p><b>System Status:</b></p>
            <p>ASTAP: {astap_status}</p>

            <p><b>Features:</b></p>
            <ul>
            <li>Advanced calibration pipeline</li>
            <li>ASTAP plate solving integration</li>
            <li>Statistical transient detection</li>
            <li>GAIA DR3 catalog matching</li>
            <li>NASA Exoplanet Archive integration</li>
            <li>Global data sharing network</li>
            </ul>

            <p>© 2025 Kelsi Davis - GeekAstro Development</p>
            <p><a href="https://geekastro.dev">https://geekastro.dev</a></p>
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
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())

        # Log application shutdown
        self.logger.info("PulseHunter application closing")
        self.add_system_log("Application shutting down...")

        # Accept the close event
        event.accept()

def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("PulseHunter")
    app.setApplicationVersion("Alpha-Enhanced")
    app.setOrganizationName("GeekAstro")
    app.setOrganizationDomain("geekastro.dev")

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = PulseHunterMainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
