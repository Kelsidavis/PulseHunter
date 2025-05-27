import csv
import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)


def create_master_frame(folder, kind="dark"):
    """Create master calibration frame from folder of FITS files"""
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
    """Interactive dialog to create master frame"""
    folder = QFileDialog.getExistingDirectory(
        None, f"Select {kind.capitalize()} Frame Folder"
    )
    if not folder:
        QMessageBox.warning(
            None,
            "No Folder Selected",
            f"You must select a folder containing {kind} frames.",
        )
        return None
    try:
        master = create_master_frame(folder, kind)
        QMessageBox.information(
            None,
            f"{kind.capitalize()} Master Created",
            f"Master {kind} created from {folder}.",
        )
        return master
    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        return None


def generate_dataset_id(folder):
    """Generate unique dataset ID from folder path and FITS files"""
    try:
        fits_files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith(".fits")]
        )
        id_string = folder + "".join(fits_files)
        return hashlib.sha256(id_string.encode()).hexdigest()
    except Exception as e:
        print(f"Dataset ID generation failed: {e}")
        return None


def generate_lightcurve_outputs(
    detections, output_folder, dataset_id="unknown", observer="Unknown"
):
    """Generate light curve CSV files and plots for all detections"""
    os.makedirs(output_folder, exist_ok=True)

    for i, det in enumerate(detections):
        if "light_curve" in det:
            lc = det["light_curve"]

            # Save CSV file
            csv_path = os.path.join(output_folder, f"lightcurve_{i:04}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Brightness"])
                for j, value in enumerate(lc):
                    writer.writerow([j, value])

            # Generate plot
            plot_path = os.path.join(output_folder, f"lightcurve_{i:04d}.png")
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(lc)), lc, marker="o", linewidth=2, markersize=4)
            plt.title(
                f"Light Curve {i}\nRA: {det.get('ra_deg', 'N/A')} "
                f"Dec: {det.get('dec_deg', 'N/A')}"
            )
            plt.xlabel("Frame Number")
            plt.ylabel("Brightness (ADU)")
            plt.grid(True, alpha=0.3)

            # Add info text
            info = det.get("match_name", "Unmatched")
            if det.get("g_mag"):
                info += f" (G={det['g_mag']:.1f})"
            if det.get("exo_match"):
                info += f" | Exoplanet: {det['exo_match']['planet']}"
            plt.figtext(0.5, 0.02, info, ha="center", fontsize=8)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

    # Create summary report
    create_summary_report(detections, output_folder, dataset_id, observer)


def create_summary_report(detections, output_folder, dataset_id, observer):
    """Create comprehensive summary report and ZIP package"""
    # Create README
    readme_path = os.path.join(output_folder, "README.txt")
    with open(readme_path, "w") as f:
        f.write("PulseHunter Detection Report\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Dataset ID: {dataset_id}\n")
        f.write(f"Total Detections: {len(detections)}\n")
        f.write(f"Observer: {observer}\n")
        f.write("Generated light curves and plots are included.\n\n")

        # Observation time range
        timestamps = [
            det.get("timestamp_utc") for det in detections if det.get("timestamp_utc")
        ]
        if timestamps:
            f.write("Observation Time Range (UTC):\n")
            f.write(f" - Start: {min(timestamps)}\n")
            f.write(f" - End: {max(timestamps)}\n\n")

        # Detection summary
        dimming_count = sum(1 for det in detections if det.get("dimming"))
        matched_count = sum(1 for det in detections if det.get("match_name"))
        exo_count = sum(1 for det in detections if det.get("exo_match"))

        f.write("Detection Summary:\n")
        f.write(f" - Brightening events: {len(detections) - dimming_count}\n")
        f.write(f" - Dimming events: {dimming_count}\n")
        f.write(f" - GAIA matches: {matched_count}\n")
        f.write(f" - Exoplanet candidates: {exo_count}\n")

    # Create JSON summary
    summary_path = os.path.join(output_folder, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_id": dataset_id,
                "observer": observer,
                "total_detections": len(detections),
                "detections": detections,
            },
            f,
            indent=2,
        )


def open_calibration_dialog():
    """Main calibration setup dialog"""
    dialog = QDialog()
    dialog.setWindowTitle("PulseHunter Calibration Setup")
    dialog.setMinimumWidth(500)
    layout = QVBoxLayout()

    # Camera type selection
    layout.addWidget(QLabel("<b>Camera Configuration:</b>"))
    camera_group = QButtonGroup(dialog)
    mono_button = QRadioButton("Monochrome Camera")
    osc_button = QRadioButton("One Shot Color (OSC) Camera")
    mono_button.setChecked(True)
    camera_group.addButton(mono_button)
    camera_group.addButton(osc_button)
    layout.addWidget(mono_button)
    layout.addWidget(osc_button)

    # Observer information
    layout.addWidget(QLabel("<b>Observer Information:</b>"))
    observer_input = QLineEdit()
    observer_input.setPlaceholderText("Enter your name or observatory")
    layout.addWidget(observer_input)

    # ASTAP configuration
    layout.addWidget(QLabel("<b>Plate Solving:</b>"))
    astap_button = QPushButton("Select ASTAP Executable")
    astap_path = ["astap"]  # Default
    astap_label = QLabel("Using default 'astap' command")

    def set_astap():
        path, _ = QFileDialog.getOpenFileName(
            None, "Select ASTAP Executable", "", "Executable files (*)"
        )
        if path:
            astap_path[0] = path
            astap_label.setText(f"Using: {os.path.basename(path)}")
            QMessageBox.information(
                None, "ASTAP Selected", f"ASTAP path set to: {path}"
            )

    astap_button.clicked.connect(set_astap)
    layout.addWidget(astap_button)
    layout.addWidget(astap_label)

    # Master frame creation
    layout.addWidget(QLabel("<b>Calibration Frames:</b>"))
    masters = {"bias": None, "dark": None, "flat": None}
    status_labels = {}

    def create_master_button(frame_type):
        button = QPushButton(f"Create {frame_type.capitalize()} Master")
        status_label = QLabel("Not created")
        status_labels[frame_type] = status_label

        def load_master():
            result = prompt_master_frame(frame_type)
            if result is not None:
                masters[frame_type] = result
                status_labels[frame_type].setText("✓ Created")
                status_labels[frame_type].setStyleSheet("color: green;")

        button.clicked.connect(load_master)
        return button, status_label

    for frame_type in ["bias", "dark", "flat"]:
        button, label = create_master_button(frame_type)
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(button)
        frame_layout.addWidget(label)
        layout.addLayout(frame_layout)

    # Dialog buttons
    button_layout = QHBoxLayout()
    confirm = QPushButton("Apply Calibration")
    cancel = QPushButton("Cancel")
    confirm.clicked.connect(dialog.accept)
    cancel.clicked.connect(dialog.reject)
    button_layout.addWidget(confirm)
    button_layout.addWidget(cancel)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if result == QDialog.Accepted:
        # Get dataset folder
        folder = QFileDialog.getExistingDirectory(None, "Select FITS Dataset Folder")
        if not folder:
            QMessageBox.warning(
                None,
                "No Folder Selected",
                "Please select your observation FITS folder.",
            )
            return None

        # Generate dataset ID
        dataset_id = generate_dataset_id(folder)
        if not dataset_id:
            QMessageBox.warning(
                None, "Dataset ID Error", "Could not generate dataset ID."
            )
            return None

        camera_type = "mono" if mono_button.isChecked() else "osc"
        observer_name = observer_input.text().strip() or "Unknown"

        return {
            "observer": observer_name,
            "astap": astap_path[0],
            "dataset_id": dataset_id,
            "dataset_folder": folder,
            "bias": masters["bias"],
            "dark": masters["dark"],
            "flat": masters["flat"],
            "camera_mode": camera_type,
        }

    return None
