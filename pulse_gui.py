import sys
import os
import json
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLineEdit, QMessageBox, QProgressBar, QCheckBox, QTextEdit
)
from PySide6.QtGui import QPixmap, QImage, QMovie
from PySide6.QtCore import Qt

from pulsehunter_core import (
    load_fits_stack, detect_transients, crossmatch_with_gaia, save_report
)
from calibration import open_calibration_dialog, generate_lightcurve_outputs
from exoplanet_match import match_transits_with_exoplanets

class PulseHunterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PulseHunter: Optical SETI Transient Detector")
        self.resize(900, 700)

        self.fits_folder = ""
        self.detections = []
        self.calibration_data = {"bias": None, "dark": None, "flat": None, "camera_mode": "mono", "astap": "astap"}

        self.folder_label = QLabel("No folder selected")
        self.select_button = QPushButton("Select FITS Folder")
        self.threshold_input = QLineEdit("6.0")
        self.calibration_button = QPushButton("Calibration Setup")
        self.run_button = QPushButton("Run Detection")
        self.filter_checkbox = QCheckBox("Show only unmatched detections")
        self.progress = QProgressBar()
        self.spinner = QLabel()
        self.spinner_movie = QMovie("/usr/share/icons/breeze/animations/process-working.gif")
        self.spinner.setMovie(self.spinner_movie)
        self.spinner.setVisible(False)

        self.detection_list = QListWidget()
        self.image_preview = QLabel("No image")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.export_button = QPushButton("Upload Report")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        layout = QVBoxLayout()
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.select_button)
        top_bar.addWidget(QLabel("Threshold:"))
        top_bar.addWidget(self.threshold_input)
        top_bar.addWidget(self.calibration_button)
        top_bar.addWidget(self.run_button)
        layout.addLayout(top_bar)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.filter_checkbox)
        layout.addWidget(self.progress)
        layout.addWidget(self.spinner)
        layout.addWidget(self.log_output)

        middle = QHBoxLayout()
        middle.addWidget(self.detection_list, 2)
        middle.addWidget(self.image_preview, 3)
        layout.addLayout(middle)

        layout.addWidget(self.export_button)

        about_label = QLabel("<b>PulseHunter</b> is an open-source Optical SETI detection tool for amateur astronomers.<br>Developed by <a href='https://geekastro.dev'>Kelsi Davis</a>.")
        about_label.setTextFormat(Qt.RichText)
        about_label.setWordWrap(True)
        about_label.setOpenExternalLinks(True)
        about_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(about_label)

        self.setLayout(layout)

        self.select_button.clicked.connect(self.select_folder)
        self.calibration_button.clicked.connect(self.open_calibration)
        self.run_button.clicked.connect(self.run_detection)
        self.detection_list.currentItemChanged.connect(self.show_preview)
        self.export_button.clicked.connect(self.export_report)
        self.filter_checkbox.stateChanged.connect(self.update_display_list)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select FITS Folder")
        if folder:
            self.fits_folder = folder
            self.folder_label.setText(f"Selected: {folder}")

    def open_calibration(self):
        result = open_calibration_dialog()
        if result:
            self.calibration_data = result
            QMessageBox.information(self, "Calibration Ready", f"Calibration loaded.\nMode: {result['camera_mode']}")

    def run_detection(self):
        if not self.fits_folder:
            QMessageBox.warning(self, "No folder", "Please select a folder first.")
            return

        try:
            z_thresh = float(self.threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid threshold", "Please enter a numeric threshold.")
            return

        self.progress.setValue(0)
        self.log_output.clear()
        self.spinner.setVisible(True)
        self.spinner_movie.start()
        self.log_output.append("Starting detection run...")

        frames, filenames, wcs_objs = load_fits_stack(
            self.fits_folder,
            plate_solve_missing=True,
            astap_exe=self.calibration_data["astap"],
            master_bias=self.calibration_data["bias"],
            master_dark=self.calibration_data["dark"],
            master_flat=self.calibration_data["flat"],
            camera_mode=self.calibration_data["camera_mode"]
        )

        self.detections = detect_transients(frames, filenames, wcs_objs, z_thresh=z_thresh, detect_dimming=True)
        self.detections = crossmatch_with_gaia(self.detections)
        self.detections = match_transits_with_exoplanets(self.detections)

        output_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Light Curves and Report")
        if not output_folder:
            QMessageBox.warning(self, "No Folder Selected", "Report saving cancelled.")
            return
        generate_lightcurve_outputs(self.detections, output_folder)

        self.spinner_movie.stop()
        self.spinner.setVisible(False)
        self.progress.setValue(100)
        self.log_output.append(f"Detection run complete. {len(self.detections)} detections found.")
        self.update_display_list()

    def update_display_list(self):
        self.detection_list.clear()
        show_only_unmatched = self.filter_checkbox.isChecked()

        for det in self.detections:
            if show_only_unmatched and det.get("match_name"):
                continue
            label = f"Frame {det['frame']} | RA: {det['ra_deg']} Dec: {det['dec_deg']} | Time: {det.get('timestamp_utc', '?')}"
            if det.get("note"):
                label += f" | {det['note']}"
            if det.get("match_name"):
                label += f" | Match: {det['match_name']} ({det['object_type']})"
                if 'g_mag' in det:
                    label += f" G={det['g_mag']:.1f}"
            if det.get("dimming"):
                label += " | 🌑 Dimming"
                if det.get("exo_match"):
                    exo = det["exo_match"]
                    label += f" | Transit Match: {exo['planet']} ({exo['host']})"
                    if exo.get("period_days"):
                        label += f", {exo['period_days']}d"
                    if exo.get("depth_ppm"):
                        label += f", {exo['depth_ppm']}ppm"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, det)
            self.detection_list.addItem(item)

    def show_preview(self, current: QListWidgetItem):
        if not current:
            self.image_preview.setText("No image")
            return

        det = current.data(Qt.UserRole)
        path = det["cutout_image"]
        if os.path.exists(path):
            img = QImage(path)
            self.image_preview.setPixmap(QPixmap.fromImage(img).scaled(
                self.image_preview.width(), self.image_preview.height(), Qt.KeepAspectRatio))
        else:
            self.image_preview.setText("Image not found")

    def export_report(self):
        if not self.detections:
            QMessageBox.information(self, "No detections", "Run detection first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "pulse_report.json")
        if out_path:
            save_report(self.detections, out_path)
            QMessageBox.information(self, "Uploaded", f"Report saved and uploaded to {out_path}")

    @staticmethod
    def get_dataset_id_from_folder(folder):
        try:
            fits_files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(".fits")
            ])
            id_string = folder + "".join(fits_files)
            return hashlib.sha256(id_string.encode()).hexdigest()
        except Exception as e:
            print("Dataset ID generation failed:", e)
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PulseHunterGUI()
    gui.show()
    sys.exit(app.exec())

