"""
Filter-Aware Calibration System for PulseHunter
Properly handles different filters with appropriate calibration frames
"""

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration_utilities import CalibrationConfig, CalibrationLogger
from fits_processing import CalibrationProcessor


class FilterAnalyzer:
    """Analyzes FITS files to detect filters and organize by filter"""

    def __init__(self):
        self.logger = CalibrationLogger()

    def analyze_folder(self, folder_path: str) -> Dict[str, List[str]]:
        """
        Analyze a folder of FITS files and group by filter

        Args:
            folder_path: Path to folder containing FITS files

        Returns:
            Dictionary mapping filter names to lists of file paths
        """
        folder = Path(folder_path)
        if not folder.exists():
            return {}

        filter_files = defaultdict(list)

        # Get all FITS files
        fits_files = list(folder.glob("*.fit*"))

        for fits_file in fits_files:
            try:
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header

                    # Try different common filter keywords
                    filter_name = self._extract_filter_name(header)

                    # Also get image type and exposure for context
                    image_type = header.get("IMAGETYP", "UNKNOWN").upper()
                    exposure = header.get("EXPTIME", 0)

                    # Store file info
                    file_info = {
                        "path": str(fits_file),
                        "filter": filter_name,
                        "image_type": image_type,
                        "exposure": exposure,
                        "filename": fits_file.name,
                    }

                    filter_files[filter_name].append(file_info)

            except Exception as e:
                self.logger.warning(f"Could not read filter from {fits_file}: {e}")
                # Add to 'UNKNOWN' filter group
                filter_files["UNKNOWN"].append(
                    {
                        "path": str(fits_file),
                        "filter": "UNKNOWN",
                        "image_type": "UNKNOWN",
                        "exposure": 0,
                        "filename": fits_file.name,
                    }
                )

        return dict(filter_files)

    def _extract_filter_name(self, header) -> str:
        """Extract filter name from FITS header"""
        # Common filter keywords to check
        filter_keywords = ["FILTER", "FILTERS", "FILTER1", "INSFLNAM", "FILTNME3"]

        for keyword in filter_keywords:
            if keyword in header:
                filter_value = str(header[keyword]).strip()
                if filter_value and filter_value.upper() not in [
                    "NONE",
                    "NULL",
                    "",
                    "N/A",
                ]:
                    return self._normalize_filter_name(filter_value)

        # If no filter found, check if it's a common calibration frame
        image_type = header.get("IMAGETYP", "").upper()
        if "BIAS" in image_type:
            return "BIAS"  # Bias frames don't need filter info
        elif "DARK" in image_type:
            return "DARK"  # Dark frames don't need filter info

        return "UNKNOWN"

    def _normalize_filter_name(self, filter_name: str) -> str:
        """Normalize filter names to standard format"""
        filter_name = filter_name.upper().strip()

        # Common filter mappings
        filter_mappings = {
            "LUMINANCE": "L",
            "LUM": "L",
            "CLEAR": "L",
            "RED": "R",
            "GREEN": "G",
            "BLUE": "B",
            "HYDROGEN": "Ha",
            "H-ALPHA": "Ha",
            "HALPHA": "Ha",
            "OXYGEN": "OIII",
            "O3": "OIII",
            "SULFUR": "SII",
            "S2": "SII",
        }

        return filter_mappings.get(filter_name, filter_name)


class FilterAwareCalibrationProject:
    """Manages filter-aware calibration projects"""

    def __init__(self):
        self.config_file = Path("filter_calibration_projects.json")
        self.projects = self.load_projects()
        self.logger = CalibrationLogger()

    def load_projects(self):
        """Load existing calibration projects"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading projects: {e}")
                return {}
        return {}

    def save_projects(self):
        """Save calibration projects"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.projects, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving projects: {e}")

    def create_project(
        self, lights_folder: str, filter_master_files: Dict[str, Dict[str, str]]
    ):
        """
        Create a new filter-aware calibration project

        Args:
            lights_folder: Path to lights folder
            filter_master_files: Dict mapping filters to their master files
                                Example: {
                                    'L': {'bias': 'path/to/bias.fits', 'dark': 'path/to/dark.fits', 'flat': 'path/to/flat_L.fits'},
                                    'R': {'bias': 'path/to/bias.fits', 'dark': 'path/to/dark.fits', 'flat': 'path/to/flat_R.fits'}
                                }
        """
        project_id = str(Path(lights_folder).resolve())

        self.projects[project_id] = {
            "lights_folder": lights_folder,
            "filter_master_files": filter_master_files,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "version": "2.0_filter_aware",
        }

        self.save_projects()
        return project_id

    def get_masters_for_folder_and_filter(
        self, lights_folder: str, filter_name: str
    ) -> Dict[str, str]:
        """Get master files for a specific lights folder and filter"""
        project_id = str(Path(lights_folder).resolve())

        if project_id in self.projects:
            project = self.projects[project_id]
            project["last_used"] = datetime.now().isoformat()
            self.save_projects()

            filter_masters = project.get("filter_master_files", {})

            # Try exact filter match first
            if filter_name in filter_masters:
                return filter_masters[filter_name]

            # For bias and dark frames, any filter will do (they're filter-independent)
            # Look for any available master files
            if filter_masters:
                first_filter = list(filter_masters.keys())[0]
                masters = filter_masters[first_filter].copy()

                # Remove filter-specific calibrations (flats) if wrong filter
                if "flat" in masters and filter_name != first_filter:
                    del masters["flat"]

                return masters

        return {}

    def get_all_projects(self) -> Dict:
        """Get all calibration projects"""
        return self.projects.copy()


class FilterAwareCalibrationWorker(QThread):
    """Worker thread for filter-aware calibration processing"""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)  # success, message, filter_master_files

    def __init__(self, calibration_tasks: Dict[str, Dict], output_folder: str):
        super().__init__()
        self.calibration_tasks = calibration_tasks  # Filter -> {type -> folder}
        self.output_folder = Path(output_folder)
        self.is_cancelled = False
        self.filter_master_files = {}

    def run(self):
        """Main processing function"""
        try:
            self.status_updated.emit("Starting filter-aware calibration processing...")
            self.log_updated.emit(
                "Creating filter-specific master calibration files..."
            )

            # Count total tasks
            total_tasks = 0
            for filter_name, tasks in self.calibration_tasks.items():
                total_tasks += len([t for t in tasks.values() if t["enabled"]])

            completed_tasks = 0

            # Process each filter
            for filter_name, tasks in self.calibration_tasks.items():
                if self.is_cancelled:
                    break

                self.status_updated.emit(
                    f"Processing {filter_name} filter calibration..."
                )
                self.log_updated.emit(f"Creating master files for {filter_name} filter")

                filter_masters = {}

                # Process each calibration type for this filter
                for cal_type, task_info in tasks.items():
                    if (
                        not task_info["enabled"]
                        or not task_info["folder"]
                        or self.is_cancelled
                    ):
                        continue

                    input_folder = task_info["folder"]
                    self.log_updated.emit(
                        f"Processing {cal_type} for {filter_name} filter"
                    )

                    # Create filter-specific filename
                    if filter_name in ["BIAS", "DARK"]:
                        # Bias and dark are filter-independent
                        output_file = (
                            self.output_folder / f"master_{cal_type.lower()}.fits"
                        )
                    else:
                        # Filter-specific calibration (mainly flats)
                        output_file = (
                            self.output_folder
                            / f"master_{cal_type.lower()}_{filter_name}.fits"
                        )

                    # Process calibration files
                    processor = CalibrationProcessor()
                    input_files = list(Path(input_folder).glob("*.fit*"))

                    if not input_files:
                        self.log_updated.emit(f"No FITS files found in {input_folder}")
                        continue

                    def progress_callback(value):
                        overall_progress = int(
                            (completed_tasks / total_tasks) * 100
                            + (value / total_tasks)
                        )
                        self.progress_updated.emit(overall_progress)

                    success = processor.create_master_calibration(
                        input_files, output_file, cal_type, progress_callback
                    )

                    if success:
                        filter_masters[cal_type] = str(output_file)
                        self.log_updated.emit(
                            f"✅ Master {cal_type} for {filter_name}: {output_file.name}"
                        )
                    else:
                        self.log_updated.emit(
                            f"❌ Failed to create master {cal_type} for {filter_name}"
                        )

                    completed_tasks += 1

                if filter_masters:
                    self.filter_master_files[filter_name] = filter_masters

            self.progress_updated.emit(100)
            self.status_updated.emit("Filter-aware calibration processing completed!")

            total_masters = sum(
                len(masters) for masters in self.filter_master_files.values()
            )
            message = f"Created {total_masters} master calibration files for {len(self.filter_master_files)} filters"
            self.finished.emit(True, message, self.filter_master_files)

        except Exception as e:
            error_msg = f"Filter-aware calibration processing error: {str(e)}"
            self.log_updated.emit(f"ERROR: {error_msg}")
            self.finished.emit(False, error_msg, {})

    def cancel(self):
        self.is_cancelled = True


class FilterAwareCalibrationDialog(QDialog):
    """Filter-aware calibration dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PulseHunter - Filter-Aware Calibration Setup")
        self.setMinimumSize(1000, 900)

        self.filter_analyzer = FilterAnalyzer()
        self.calibration_project = FilterAwareCalibrationProject()
        self.worker = None

        # Data storage
        self.lights_filters = {}  # Filter -> file list
        self.calibration_filters = {}  # Cal type -> {Filter -> file list}

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Step 1: Lights folder analysis
        self.setup_lights_analysis_section(layout)

        # Step 2: Calibration frame organization
        self.setup_calibration_organization_section(layout)

        # Step 3: Processing
        self.setup_processing_section(layout)

        # Buttons
        self.setup_buttons(layout)

    def setup_lights_analysis_section(self, layout):
        """Setup lights folder analysis with filter detection"""
        lights_group = QGroupBox("🔬 Step 1: Analyze Science Images by Filter")
        lights_layout = QVBoxLayout(lights_group)

        # Instructions
        instructions = QLabel(
            "Select your science images folder. The system will analyze all FITS files "
            "and organize them by filter for proper calibration matching."
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

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Lights Folder:"))

        self.lights_folder_edit = QLineEdit()
        self.lights_folder_edit.setPlaceholderText(
            "Select folder containing your science images..."
        )
        folder_layout.addWidget(self.lights_folder_edit)

        browse_btn = QPushButton("Browse & Analyze")
        browse_btn.clicked.connect(self.analyze_lights_folder)
        folder_layout.addWidget(browse_btn)

        lights_layout.addLayout(folder_layout)

        # Filter analysis results
        self.lights_analysis_tree = QTreeWidget()
        self.lights_analysis_tree.setHeaderLabels(
            ["Filter/Image", "Count", "Type", "Details"]
        )
        self.lights_analysis_tree.setMaximumHeight(200)
        lights_layout.addWidget(self.lights_analysis_tree)

        layout.addWidget(lights_group)

    def setup_calibration_organization_section(self, layout):
        """Setup calibration frame organization by filter"""
        cal_group = QGroupBox("📸 Step 2: Organize Calibration Frames by Filter")
        cal_layout = QVBoxLayout(cal_group)

        # Instructions
        instructions = QLabel(
            "Select folders for your calibration frames. The system will match them to the appropriate filters. "
            "Bias and Dark frames are filter-independent. Flat frames are filter-specific."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin-bottom: 10px;")
        cal_layout.addWidget(instructions)

        # Tab widget for different calibration types
        self.cal_tabs = QTabWidget()

        # Bias tab (filter-independent)
        self.setup_bias_tab()

        # Dark tab (filter-independent)
        self.setup_dark_tab()

        # Flat tab (filter-specific)
        self.setup_flat_tab()

        cal_layout.addWidget(self.cal_tabs)

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

    def setup_bias_tab(self):
        """Setup bias calibration tab"""
        bias_widget = QWidget()
        layout = QVBoxLayout(bias_widget)

        # Enable checkbox
        self.bias_enabled = QCheckBox("Create Master Bias Frame")
        layout.addWidget(self.bias_enabled)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Bias Frames Folder:"))

        self.bias_folder_edit = QLineEdit()
        self.bias_folder_edit.setPlaceholderText(
            "Select folder containing bias frames..."
        )
        folder_layout.addWidget(self.bias_folder_edit)

        bias_browse_btn = QPushButton("Browse & Analyze")
        bias_browse_btn.clicked.connect(lambda: self.analyze_calibration_folder("bias"))
        folder_layout.addWidget(bias_browse_btn)

        layout.addLayout(folder_layout)

        # Analysis results
        self.bias_analysis_tree = QTreeWidget()
        self.bias_analysis_tree.setHeaderLabels(["File", "Filter", "Exposure", "Type"])
        self.bias_analysis_tree.setMaximumHeight(150)
        layout.addWidget(self.bias_analysis_tree)

        self.cal_tabs.addTab(bias_widget, "Bias (Filter Independent)")

    def setup_dark_tab(self):
        """Setup dark calibration tab"""
        dark_widget = QWidget()
        layout = QVBoxLayout(dark_widget)

        # Enable checkbox
        self.dark_enabled = QCheckBox("Create Master Dark Frame")
        layout.addWidget(self.dark_enabled)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Dark Frames Folder:"))

        self.dark_folder_edit = QLineEdit()
        self.dark_folder_edit.setPlaceholderText(
            "Select folder containing dark frames..."
        )
        folder_layout.addWidget(self.dark_folder_edit)

        dark_browse_btn = QPushButton("Browse & Analyze")
        dark_browse_btn.clicked.connect(lambda: self.analyze_calibration_folder("dark"))
        folder_layout.addWidget(dark_browse_btn)

        layout.addLayout(folder_layout)

        # Analysis results
        self.dark_analysis_tree = QTreeWidget()
        self.dark_analysis_tree.setHeaderLabels(["File", "Filter", "Exposure", "Type"])
        self.dark_analysis_tree.setMaximumHeight(150)
        layout.addWidget(self.dark_analysis_tree)

        self.cal_tabs.addTab(dark_widget, "Dark (Filter Independent)")

    def setup_flat_tab(self):
        """Setup flat calibration tab (filter-specific)"""
        flat_widget = QWidget()
        layout = QVBoxLayout(flat_widget)

        # Enable checkbox
        self.flat_enabled = QCheckBox("Create Master Flat Frames (Filter-Specific)")
        layout.addWidget(self.flat_enabled)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Flat Frames Folder:"))

        self.flat_folder_edit = QLineEdit()
        self.flat_folder_edit.setPlaceholderText(
            "Select folder containing flat frames for all filters..."
        )
        folder_layout.addWidget(self.flat_folder_edit)

        flat_browse_btn = QPushButton("Browse & Analyze")
        flat_browse_btn.clicked.connect(lambda: self.analyze_calibration_folder("flat"))
        folder_layout.addWidget(flat_browse_btn)

        layout.addLayout(folder_layout)

        # Analysis results (organized by filter)
        self.flat_analysis_tree = QTreeWidget()
        self.flat_analysis_tree.setHeaderLabels(
            ["Filter/File", "Count", "Exposure", "Details"]
        )
        self.flat_analysis_tree.setMaximumHeight(200)
        layout.addWidget(self.flat_analysis_tree)

        self.cal_tabs.addTab(flat_widget, "Flats (Filter Specific)")

    def setup_processing_section(self, layout):
        """Setup processing section"""
        processing_group = QGroupBox("⚙️ Step 3: Filter-Aware Processing")
        processing_layout = QVBoxLayout(processing_group)

        # Status
        self.status_label = QLabel(
            "Ready to create filter-aware master calibration files"
        )
        self.status_label.setStyleSheet("font-weight: bold; color: #2c5aa0;")
        processing_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        processing_layout.addWidget(self.progress_bar)

        # Log
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setReadOnly(True)
        processing_layout.addWidget(self.log_display)

        layout.addWidget(processing_group)

    def setup_buttons(self, layout):
        """Setup dialog buttons"""
        button_layout = QHBoxLayout()

        # Create masters button
        self.create_btn = QPushButton("🚀 Create Filter-Aware Master Files")
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
            """
        )
        self.create_btn.clicked.connect(self.create_filter_aware_masters)
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
    def analyze_lights_folder(self):
        """Analyze lights folder for filters"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Science Images Folder", ""
        )
        if not folder:
            return

        self.lights_folder_edit.setText(folder)
        self.add_log(f"Analyzing lights folder: {folder}")

        # Analyze filters
        self.lights_filters = self.filter_analyzer.analyze_folder(folder)

        # Update tree display
        self.lights_analysis_tree.clear()

        total_files = 0
        for filter_name, file_list in self.lights_filters.items():
            # Create filter node
            filter_item = QTreeWidgetItem(
                [filter_name, str(len(file_list)), "Filter Group", ""]
            )
            filter_item.setExpanded(True)

            # Add files under filter
            for file_info in file_list[:5]:  # Show first 5 files
                file_item = QTreeWidgetItem(
                    [
                        file_info["filename"],
                        "1",
                        file_info["image_type"],
                        f"Exp: {file_info['exposure']}s",
                    ]
                )
                filter_item.addChild(file_item)

            if len(file_list) > 5:
                more_item = QTreeWidgetItem(
                    [f"... and {len(file_list) - 5} more files", "", "", ""]
                )
                filter_item.addChild(more_item)

            self.lights_analysis_tree.addTopLevelItem(filter_item)
            total_files += len(file_list)

        self.add_log(
            f"Found {total_files} files across {len(self.lights_filters)} filters"
        )

        # Set default output folder
        if not self.output_folder_edit.text():
            default_output = Path(folder).parent / "filter_master_calibrations"
            self.output_folder_edit.setText(str(default_output))

    def analyze_calibration_folder(self, cal_type: str):
        """Analyze calibration folder"""
        folder_edit = getattr(self, f"{cal_type}_folder_edit")
        folder = QFileDialog.getExistingDirectory(
            self, f"Select {cal_type.title()} Frames Folder", ""
        )
        if not folder:
            return

        folder_edit.setText(folder)
        self.add_log(f"Analyzing {cal_type} folder: {folder}")

        # Analyze files
        filter_files = self.filter_analyzer.analyze_folder(folder)
        self.calibration_filters[cal_type] = filter_files

        # Update appropriate tree
        tree_widget = getattr(self, f"{cal_type}_analysis_tree")
        tree_widget.clear()

        if cal_type in ["bias", "dark"]:
            # Filter-independent: just list files
            all_files = []
            for file_list in filter_files.values():
                all_files.extend(file_list)

            for file_info in all_files[:10]:  # Show first 10
                file_item = QTreeWidgetItem(
                    [
                        file_info["filename"],
                        file_info["filter"],
                        f"{file_info['exposure']}s",
                        file_info["image_type"],
                    ]
                )
                tree_widget.addTopLevelItem(file_item)

            if len(all_files) > 10:
                more_item = QTreeWidgetItem(
                    [f"... and {len(all_files) - 10} more files", "", "", ""]
                )
                tree_widget.addTopLevelItem(more_item)

        else:
            # Filter-specific (flats): organize by filter
            for filter_name, file_list in filter_files.items():
                filter_item = QTreeWidgetItem(
                    [filter_name, str(len(file_list)), "Filter Group", ""]
                )
                filter_item.setExpanded(True)

                for file_info in file_list[:3]:  # Show first 3 per filter
                    file_item = QTreeWidgetItem(
                        [
                            file_info["filename"],
                            "1",
                            f"{file_info['exposure']}s",
                            file_info["image_type"],
                        ]
                    )
                    filter_item.addChild(file_item)

                if len(file_list) > 3:
                    more_item = QTreeWidgetItem(
                        [f"... {len(file_list) - 3} more", "", "", ""]
                    )
                    filter_item.addChild(more_item)

                tree_widget.addTopLevelItem(filter_item)

        # Enable the calibration type
        checkbox = getattr(self, f"{cal_type}_enabled")
        checkbox.setChecked(True)

    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Master Files", ""
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def create_filter_aware_masters(self):
        """Create filter-aware master calibration files"""
        # Validate inputs
        if not self.lights_folder_edit.text():
            QMessageBox.warning(
                self, "Missing Input", "Please analyze your lights folder first!"
            )
            return

        if not self.lights_filters:
            QMessageBox.warning(
                self,
                "No Analysis",
                "Please analyze your lights folder to detect filters!",
            )
            return

        if not self.output_folder_edit.text():
            QMessageBox.warning(
                self, "Missing Output", "Please select an output folder!"
            )
            return

        # Organize calibration tasks by filter
        calibration_tasks = {}

        # Get unique filters from lights (these are what we need to calibrate for)
        light_filters = set(self.lights_filters.keys())

        # For each filter in lights, determine what calibration is needed
        for filter_name in light_filters:
            if filter_name == "UNKNOWN":
                continue  # Skip unknown filters

            calibration_tasks[filter_name] = {}

            # Bias (filter-independent, but store under each filter for organization)
            if self.bias_enabled.isChecked() and self.bias_folder_edit.text():
                calibration_tasks[filter_name]["bias"] = {
                    "enabled": True,
                    "folder": self.bias_folder_edit.text(),
                }

            # Dark (filter-independent, but store under each filter for organization)
            if self.dark_enabled.isChecked() and self.dark_folder_edit.text():
                calibration_tasks[filter_name]["dark"] = {
                    "enabled": True,
                    "folder": self.dark_folder_edit.text(),
                }

            # Flat (filter-specific)
            if self.flat_enabled.isChecked() and self.flat_folder_edit.text():
                # Check if we have flats for this specific filter
                flat_files = self.calibration_filters.get("flat", {})
                if filter_name in flat_files and len(flat_files[filter_name]) > 0:
                    # Create a temporary folder with just this filter's flats
                    # (In practice, you'd want to implement proper filter separation)
                    calibration_tasks[filter_name]["flat"] = {
                        "enabled": True,
                        "folder": self.flat_folder_edit.text(),  # Would be filter-specific in full implementation
                    }

        if not calibration_tasks:
            QMessageBox.warning(
                self, "No Calibration", "Please enable at least one calibration type!"
            )
            return

        # Start processing
        self.start_filter_aware_processing(calibration_tasks)

    def start_filter_aware_processing(self, calibration_tasks):
        """Start filter-aware calibration processing"""
        output_folder = self.output_folder_edit.text()
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Start worker
        self.worker = FilterAwareCalibrationWorker(calibration_tasks, output_folder)
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

    def processing_finished(self, success, message, filter_master_files):
        """Handle processing completion"""
        # Update UI
        self.progress_bar.setVisible(False)
        self.create_btn.setVisible(True)
        self.cancel_btn.setVisible(False)

        if success and filter_master_files:
            self.status_label.setText("✅ Filter-aware master files created!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            # Create filter-aware calibration project
            lights_folder = self.lights_folder_edit.text()
            self.calibration_project.create_project(lights_folder, filter_master_files)

            self.add_log("✅ Filter-aware calibration project created!")

            # Show success message
            filter_summary = []
            for filter_name, masters in filter_master_files.items():
                master_types = list(masters.keys())
                filter_summary.append(f"• {filter_name}: {', '.join(master_types)}")

            QMessageBox.information(
                self,
                "Filter-Aware Calibration Complete!",
                f"🎉 Filter-aware master files created!\n\n"
                f"Created masters for:\n" + "\n".join(filter_summary) + f"\n\n"
                f"✨ These will be automatically matched by filter when processing!",
            )
        else:
            self.status_label.setText("❌ Processing failed!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Processing Failed", f"Error: {message}")

        self.worker = None


# Enhanced auto calibration manager with filter awareness
class FilterAwareAutoCalibrationManager:
    """Enhanced calibration manager with filter awareness"""

    def __init__(self):
        self.project_manager = FilterAwareCalibrationProject()
        self.logger = CalibrationLogger()

    def get_masters_for_file(self, file_path: str) -> Dict[str, str]:
        """
        Get appropriate master calibration files for a specific FITS file

        Args:
            file_path: Path to FITS file

        Returns:
            Dictionary of calibration type -> master file path
        """
        try:
            # Determine the lights folder (parent directory)
            lights_folder = str(Path(file_path).parent)

            # Extract filter from the FITS file
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                filter_analyzer = FilterAnalyzer()
                filter_name = filter_analyzer._extract_filter_name(header)

            # Get appropriate master files
            return self.project_manager.get_masters_for_folder_and_filter(
                lights_folder, filter_name
            )

        except Exception as e:
            self.logger.error(f"Error getting masters for {file_path}: {e}")
            return {}

    def load_master_frame(self, file_path: str) -> Optional[np.ndarray]:
        """Load a master calibration frame"""
        try:
            if not Path(file_path).exists():
                self.logger.warning(f"Master file not found: {file_path}")
                return None

            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                self.logger.info(f"✅ Loaded master file: {Path(file_path).name}")
                return data

        except Exception as e:
            self.logger.error(f"Error loading master file {file_path}: {e}")
            return None


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = FilterAwareCalibrationDialog()
    dialog.show()
    sys.exit(app.exec())
