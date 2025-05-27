import hashlib
import os
import sys

from PySide6.QtCore import Signal, Qt, QThread
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calibration import generate_lightcurve_outputs, open_calibration_dialog
from exoplanet_match import match_transits_with_exoplanets
from pulsehunter_core import (
    crossmatch_with_gaia,
    detect_transients,
    load_fits_stack,
    save_report,
)


class DetectionWorker(QThread):
    """Worker thread for detection processing to prevent GUI freezing"""

    progress_updated = Signal(int)
    log_updated = Signal(str)
    detection_complete = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, fits_folder, z_thresh, calibration_data):
        super().__init__()
        self.fits_folder = fits_folder
        self.z_thresh = z_thresh
        self.calibration_data = calibration_data

    def run(self):
        try:
            self.log_updated.emit("Loading FITS files and performing calibration...")
            self.progress_updated.emit(10)

            frames, filenames, wcs_objs = load_fits_stack(
                self.fits_folder,
                plate_solve_missing=True,
                astap_exe=self.calibration_data["astap"],
                master_bias=self.calibration_data["bias"],
                master_dark=self.calibration_data["dark"],
                master_flat=self.calibration_data["flat"],
                camera_mode=self.calibration_data["camera_mode"],
            )

            self.log_updated.emit(f"Loaded {len(frames)} frames successfully")
            self.progress_updated.emit(30)

            self.log_updated.emit("Detecting transients...")
            detections = detect_transients(
                frames,
                filenames,
                wcs_objs,
                z_thresh=self.z_thresh,
                detect_dimming=True,
            )
            self.progress_updated.emit(60)

            self.log_updated.emit(f"Found {len(detections)} potential detections")

            self.log_updated.emit("Cross-matching with GAIA catalog...")
            detections = crossmatch_with_gaia(detections)
            self.progress_updated.emit(80)

            self.log_updated.emit("Checking for exoplanet transit matches...")
            detections = match_transits_with_exoplanets(detections)
            self.progress_updated.emit(100)

            self.detection_complete.emit(detections)

        except Exception as e:
            self.error_occurred.emit(str(e))


class PulseHunterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PulseHunter: Optical SETI & Exoplanet Detection")
        self.resize(1200, 800)
        self.setStyleSheet(self.get_modern_stylesheet())

        # Application state
        self.fits_folder = ""
        self.detections = []
        self.calibration_data = {
            "bias": None,
            "dark": None,
            "flat": None,
            "camera_mode": "mono",
            "astap": "astap",
            "observer": "Unknown",
            "dataset_id": None,
        }
        self.worker = None

        self.setup_ui()
        self.connect_signals()

    def get_modern_stylesheet(self):
        return """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QPushButton {
            background-color: #0078d4;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #404040;
            color: #808080;
        }
        QLineEdit {
            background-color: #404040;
            border: 1px solid #606060;
            padding: 6px;
            border-radius: 3px;
        }
        QListWidget {
            background-color: #404040;
            border: 1px solid #606060;
            selection-background-color: #0078d4;
        }
        QTextEdit {
            background-color: #1e1e1e;
            border: 1px solid #606060;
            font-family: 'Consolas', monospace;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #606060;
            margin-top: 8px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QProgressBar {
            border: 1px solid #606060;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 2px;
        }
        """

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Header
        header = QLabel("PulseHunter")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 24, QFont.Bold))
        header.setStyleSheet("color: #0078d4; margin: 10px;")
        main_layout.addWidget(header)

        subtitle = QLabel("Optical SETI & Exoplanet Transit Detection Network")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #cccccc; margin-bottom: 20px;")
        main_layout.addWidget(subtitle)

        # Control panel
        control_group = QGroupBox("Detection Controls")
        control_layout = QFormLayout()

        self.folder_label = QLabel("No folder selected")
        self.select_button = QPushButton("Select FITS Folder")

        threshold_layout = QHBoxLayout()
        self.threshold_input = QLineEdit("6.0")
        self.threshold_input.setMaximumWidth(100)
        threshold_layout.addWidget(self.threshold_input)
        threshold_layout.addWidget(QLabel("σ"))
        threshold_layout.addStretch()

        self.calibration_button = QPushButton("Calibration Setup")
        self.run_button = QPushButton("Start Detection")
        self.run_button.setEnabled(False)

        control_layout.addRow("Dataset Folder:", self.select_button)
        control_layout.addRow("", self.folder_label)
        control_layout.addRow("Detection Threshold:", threshold_layout)
        control_layout.addRow("Calibration:", self.calibration_button)
        control_layout.addRow("", self.run_button)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Progress section
        progress_group = QGroupBox("Processing Status")
        progress_layout = QVBoxLayout()

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        progress_layout.addWidget(self.progress)

        self.log_output = QTextEdit()
        self.log_output.setMaximumHeight(120)
        self.log_output.setPlaceholderText("Processing logs will appear here...")
        progress_layout.addWidget(self.log_output)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)

        # Results section
        results_splitter = QSplitter(Qt.Horizontal)

        # Detection list
        list_group = QGroupBox("Detections")
        list_layout = QVBoxLayout()

        filter_layout = QHBoxLayout()
        self.filter_checkbox = QCheckBox("Show only unmatched detections")
        self.detection_count_label = QLabel("0 detections")
        filter_layout.addWidget(self.filter_checkbox)
        filter_layout.addStretch()
        filter_layout.addWidget(self.detection_count_label)

        list_layout.addLayout(filter_layout)

        self.detection_list = QListWidget()
        list_layout.addWidget(self.detection_list)

        list_group.setLayout(list_layout)
        results_splitter.addWidget(list_group)

        # Image preview
        preview_group = QGroupBox("Detection Preview")
        preview_layout = QVBoxLayout()

        self.image_preview = QLabel("Select a detection to view preview")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(300)
        self.image_preview.setStyleSheet(
            "border: 1px solid #606060; background-color: #1e1e1e;"
        )
        preview_layout.addWidget(self.image_preview)

        preview_group.setLayout(preview_layout)
        results_splitter.addWidget(preview_group)

        results_splitter.setSizes([400, 500])
        main_layout.addWidget(results_splitter)

        # Export section
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_button = QPushButton("Generate & Upload Report")
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_button)
        main_layout.addLayout(export_layout)

        # Footer
        footer = QLabel(
            "<b>PulseHunter</b> is an open-source project. "
            'Visit <a href="https://geekastro.dev" style="color: #0078d4;">'
            "geekastro.dev</a> for more information."
        )
        footer.setTextFormat(Qt.RichText)
        footer.setOpenExternalLinks(True)
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("margin-top: 20px; color: #cccccc;")
        main_layout.addWidget(footer)

        self.setLayout(main_layout)

    def connect_signals(self):
        self.select_button.clicked.connect(self.select_folder)
        self.calibration_button.clicked.connect(self.open_calibration)
        self.run_button.clicked.connect(self.run_detection)
        self.detection_list.currentItemChanged.connect(self.show_preview)
        self.export_button.clicked.connect(self.export_report)
        self.filter_checkbox.stateChanged.connect(self.update_display_list)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select FITS Dataset Folder")
        if folder:
            self.fits_folder = folder
            self.folder_label.setText(f"Selected: {os.path.basename(folder)}")
            self.folder_label.setToolTip(folder)
            self.run_button.setEnabled(True)
            self.log_output.append(f"✓ Dataset folder selected: {folder}")

    def open_calibration(self):
        result = open_calibration_dialog()
        if result:
            self.calibration_data.update(result)
            QMessageBox.information(
                self,
                "Calibration Ready",
                f"Calibration configured successfully!\n"
                f"Observer: {result['observer']}\n"
                f"Camera Mode: {result['camera_mode']}\n"
                f"Dataset ID: {result['dataset_id'][:16]}...",
            )
            self.log_output.append(f"✓ Calibration configured for {result['observer']}")

    def run_detection(self):
        if not self.fits_folder:
            QMessageBox.warning(
                self, "No Dataset", "Please select a FITS folder first."
            )
            return

        try:
            z_thresh = float(self.threshold_input.text())
            if z_thresh <= 0:
                raise ValueError("Threshold must be positive")
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Threshold",
                "Please enter a valid positive number for threshold.",
            )
            return
        # Disable controls during processing
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_output.clear()

        # Start worker thread
        self.worker = DetectionWorker(self.fits_folder, z_thresh, self.calibration_data)
        self.worker.progress_updated.connect(self.progress.setValue)
        self.worker.log_updated.connect(self.log_output.append)
        self.worker.detection_complete.connect(self.on_detection_complete)
        self.worker.error_occurred.connect(self.on_detection_error)
        self.worker.start()

    def on_detection_complete(self, detections):
        self.detections = detections
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(True)

        self.log_output.append(
            f"✅ Detection complete! Found {len(detections)} detections"
        )
        self.update_display_list()

        # Show summary
        dimming_count = sum(1 for d in detections if d.get("dimming"))
        matched_count = sum(1 for d in detections if d.get("match_name"))
        exo_count = sum(1 for d in detections if d.get("exo_match"))

        summary = f"""Detection Summary:
• Total detections: {len(detections)}
• Brightening events: {len(detections) - dimming_count}
• Dimming events: {dimming_count}
• GAIA matches: {matched_count}
• Exoplanet candidates: {exo_count}"""

        QMessageBox.information(self, "Detection Complete", summary)

    def on_detection_error(self, error_msg):
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.log_output.append(f"❌ Error: {error_msg}")
        QMessageBox.critical(
            self,
            "Detection Failed",
            f"An error occurred during detection:\n\n{error_msg}",
        )

    def update_display_list(self):
        self.detection_list.clear()
        show_only_unmatched = self.filter_checkbox.isChecked()
        displayed_count = 0

        for i, det in enumerate(self.detections):
            if show_only_unmatched and det.get("match_name"):
                continue

            # Build detection label
            confidence_pct = int((det.get("confidence", 0) * 100))
            label = f"#{i + 1} | Frame {det['frame']} | Confidence: {confidence_pct}%"

            if det.get("ra_deg") and det.get("dec_deg"):
                label += f" | RA: {det['ra_deg']:.4f}° " f"Dec: {det['dec_deg']:.4f}°"

            if det.get("dimming"):
                label += " | 🌑 DIMMING"
                if det.get("exo_match"):
                    exo = det["exo_match"]
                    label += f" | 🪐 {exo['planet']}"
            else:
                label += " | ✨ BRIGHTENING"

            if det.get("match_name"):
                label += f" | Match: {det['match_name']}"
                if det.get("g_mag"):
                    label += f" (G={det['g_mag']:.1f})"

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, det)

            # Color code by confidence
            if confidence_pct >= 80:
                item.setBackground(Qt.darkGreen)
            elif confidence_pct >= 50:
                item.setBackground(Qt.darkYellow)
            else:
                item.setBackground(Qt.darkRed)

            self.detection_list.addItem(item)
            displayed_count += 1

        # Update count label
        total_text = f"{displayed_count} detections"
        if show_only_unmatched and displayed_count < len(self.detections):
            total_text += f" (of {len(self.detections)} total)"
        self.detection_count_label.setText(total_text)

    def show_preview(self, current: QListWidgetItem):
        if not current:
            self.image_preview.setText("Select a detection to view preview")
            return

        det = current.data(Qt.UserRole)
        cutout_path = det.get("cutout_image", "")

        if cutout_path and os.path.exists(cutout_path):
            img = QImage(cutout_path)
            if not img.isNull():
                scaled_pixmap = QPixmap.fromImage(img).scaled(
                    self.image_preview.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.image_preview.setPixmap(scaled_pixmap)
            else:
                self.image_preview.setText("Could not load image")
        else:
            self.image_preview.setText("No preview image available")

    def export_report(self):
        if not self.detections:
            QMessageBox.information(
                self, "No Detections", "Run detection analysis first."
            )
            return

        # Select output folder for light curves
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Light Curves and Report"
        )
        if not output_folder:
            return

        try:
            # Generate light curve outputs
            generate_lightcurve_outputs(
                self.detections,
                output_folder,
                self.calibration_data.get("dataset_id", "unknown"),
                self.calibration_data.get("observer", "Unknown"),
            )

            # Save main report JSON
            report_path = os.path.join(output_folder, "pulse_report.json")
            save_report(self.detections, report_path)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Report generated successfully!\n\n"
                f"• Light curves: {len(self.detections)} CSV files\n"
                f"• Plots: {len(self.detections)} PNG files\n"
                f"• Summary report: pulse_report.json\n"
                f"• README: README.txt\n\n"
                f"Files saved to: {output_folder}",
            )

        except Exception as error:
            QMessageBox.critical(
                self, "Export Failed", f"Failed to generate report:\n\n{str(error)}"
            )

    @staticmethod
    def get_dataset_id_from_folder(folder):
        """Generate dataset ID from folder path and contents"""
        try:
            fits_files = sorted(
                [f for f in os.listdir(folder) if f.lower().endswith(".fits")]
            )
            id_string = folder + "".join(fits_files)
            return hashlib.sha256(id_string.encode()).hexdigest()
        except Exception as e:
            print("Dataset ID generation failed:", e)
            return None


def main():
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("PulseHunter")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("GeekAstro")
    app.setOrganizationDomain("geekastro.dev")

    gui = PulseHunterGUI()
    gui.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
