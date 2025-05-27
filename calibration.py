import os
import json
import numpy as np
from astropy.io import fits
from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup
)

def create_master_frame(folder, kind="dark"):
    stack = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".fits"):
            try:
                path = os.path.join(folder, file)
                data = fits.getdata(path).astype(np.float32)
                stack.append(data)
            except Exception as e:
                print(f"Skipping {file}: {e}")
    if not stack:
        raise ValueError("No usable calibration frames found.")
    master = np.median(np.stack(stack), axis=0)
    print(f"Created master {kind} frame from {len(stack)} files")
    return master

def prompt_master_frame(kind):
    folder = QFileDialog.getExistingDirectory(None, f"Select {kind.capitalize()} Frame Folder")
    if not folder:
        QMessageBox.warning(None, "No Folder Selected", f"You must select a folder containing {kind} frames.")
        return None
    try:
        master = create_master_frame(folder, kind)
        QMessageBox.information(None, f"{kind.capitalize()} Master Created", f"Master {kind} created from {folder}.")
        return master
    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        return None

def open_calibration_dialog():
    dialog = QDialog()
    dialog.setWindowTitle("Calibration Setup")
    layout = QVBoxLayout()

    layout.addWidget(QLabel("Select Camera Type:"))
    camera_group = QButtonGroup(dialog)
    mono_button = QRadioButton("Monochrome")
    osc_button = QRadioButton("One Shot Color")
    mono_button.setChecked(True)
    camera_group.addButton(mono_button)
    camera_group.addButton(osc_button)
    layout.addWidget(mono_button)
    layout.addWidget(osc_button)

    layout.addWidget(QLabel("Observer Name:"))
    from PySide6.QtWidgets import QLineEdit
    observer_input = QLineEdit()
    layout.addWidget(observer_input)

    layout.addWidget(QLabel("Select ASTAP Executable:"))
    astap_button = QPushButton("Choose ASTAP Path")
    astap_path = ["astap"]

    def set_astap():
        path, _ = QFileDialog.getOpenFileName(None, "Select ASTAP Executable")
        if path:
            astap_path[0] = path
            QMessageBox.information(None, "ASTAP Selected", f"Using ASTAP: {path}")

    astap_button.clicked.connect(set_astap)
    layout.addWidget(astap_button)

    layout.addWidget(QLabel("Create or Load Master Frames:"))
    btn_bias = QPushButton("Create Bias Master")
    btn_dark = QPushButton("Create Dark Master")
    btn_flat = QPushButton("Create Flat Master")

    masters = {"bias": None, "dark": None, "flat": None}

    def load_bias():
        masters["bias"] = prompt_master_frame("bias")
    def load_dark():
        masters["dark"] = prompt_master_frame("dark")
    def load_flat():
        masters["flat"] = prompt_master_frame("flat")

    btn_bias.clicked.connect(load_bias)
    btn_dark.clicked.connect(load_dark)
    btn_flat.clicked.connect(load_flat)

    layout.addWidget(btn_bias)
    layout.addWidget(btn_dark)
    layout.addWidget(btn_flat)

    dataset_id_label = QLabel("Dataset ID will appear after folder selection.")
    layout.addWidget(dataset_id_label)

    confirm = QPushButton("Apply Calibration")
    cancel = QPushButton("Cancel")
    confirm.clicked.connect(dialog.accept)
    cancel.clicked.connect(dialog.reject)

    btn_row = QHBoxLayout()
    btn_row.addWidget(confirm)
    btn_row.addWidget(cancel)
    layout.addLayout(btn_row)

    dialog.setLayout(layout)
    result = dialog.exec()

    if result == QDialog.Accepted:
        camera_type = "mono" if mono_button.isChecked() else "osc"
                import hashlib
        folder = QFileDialog.getExistingDirectory(None, "Select FITS Dataset Folder")
        if not folder:
            QMessageBox.warning(None, "No Folder Selected", "Please select your observation FITS folder.")
            return None

        fits_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.fits')])
        id_string = folder + ''.join(fits_files)
                dataset_id = hashlib.sha256(id_string.encode()).hexdigest()
        dataset_id_label.setText(f"Dataset ID: {dataset_id[:10]}…")
                import matplotlib.pyplot as plt
        import csv
        from PySide6.QtWidgets import QFileDialog

        # Prompt for light curve output folder
        output_folder = QFileDialog.getExistingDirectory(None, "Select Folder to Save Light Curves and Plots")
        if output_folder:
            for i, det in enumerate(masters.get("detections", [])):
                if "light_curve" in det:
                    lc = det["light_curve"]
                    csv_path = os.path.join(output_folder, f"lightcurve_{i:04}.csv")
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Frame", "Brightness"])
                        for j, value in enumerate(lc):
                            writer.writerow([j, value])

                    plot_path = os.path.join(output_folder, f"lightcurve_{i:04}.png")
                    plt.figure()
                    plt.plot(range(len(lc)), lc, marker='o')
                    plt.title(f"Light Curve {i}
RA: {det.get('ra_deg')}  Dec: {det.get('dec_deg')}")
                    plt.xlabel("Frame")
                    plt.ylabel("Brightness")
                    info = det.get("match_name", "Unmatched")
                    if det.get("g_mag"):
                        info += f" (G={det['g_mag']:.1f})"
                    plt.figtext(0.5, 0.01, info, ha="center", fontsize=8)
                    plt.savefig(plot_path)
                    plt.close()

                # Optional: Prompt user to save a zipped report package
        from zipfile import ZipFile
        report_path, _ = QFileDialog.getSaveFileName(None, "Save Report Package", "pulse_report.zip")
        if report_path:
            with ZipFile(report_path, 'w') as zipf:
                # Include summary JSON
                readme_path = os.path.join(output_folder, "README.txt")
                with open(readme_path, "w") as f:
                    f.write(f"PulseHunter Report
")
                    f.write(f"Dataset ID: {dataset_id}
")
                    f.write(f"Total Detections: {len(masters.get('detections', []))}
")
                    f.write(f"Camera Type: {camera_type}
")
                    f.write("Generated light curves and plots are included.
")
                    f.write("
Observation Time Range (UTC):
")
                    timestamps = [det.get("timestamp_utc") for det in masters.get("detections", []) if det.get("timestamp_utc")]
                    if timestamps:
                        f.write(f" - Start: {min(timestamps)}
")
                        f.write(f" - End: {max(timestamps)}
")
                    observer = masters.get("observer", "Unknown")
                    f.write(f"
Observer: {observer}
")
                    f.write("
FITS Files:
")
                    for fname in fits_files:
                        f.write(f" - {fname}
")
                zipf.write(readme_path, arcname="README.txt")
                summary_path = os.path.join(output_folder, "summary.json")
                with open(summary_path, "w") as f:
                    json.dump(masters.get("detections", []), f, indent=2)
                zipf.write(summary_path, arcname="summary.json")

                for i, det in enumerate(masters.get("detections", [])):
                    base = f"lightcurve_{i:04}"
                    csv_file = os.path.join(output_folder, f"{base}.csv")
                    png_file = os.path.join(output_folder, f"{base}.png")
                    if os.path.exists(csv_file):
                        zipf.write(csv_file, arcname=os.path.basename(csv_file))
                    if os.path.exists(png_file):
                        zipf.write(png_file, arcname=os.path.basename(png_file))

                observer_name = observer_input.text().strip() or "Unknown"
        return {
            "observer": observer_name,
            "astap": astap_path[0],
            "dataset_id": dataset_id,
            "bias": masters["bias"],
            "dark": masters["dark"],
            "flat": masters["flat"],
            "camera_mode": camera_type
        }
    return None

