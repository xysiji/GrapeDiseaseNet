from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QUrl


class MainWindow(QMainWindow):
    open_image_requested = pyqtSignal()
    open_weights_requested = pyqtSignal()
    open_output_dir_requested = pyqtSignal()
    import_weights_requested = pyqtSignal()
    refresh_models_requested = pyqtSignal()
    registered_model_changed = pyqtSignal(str)
    predict_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Grape Leaf Disease Detection System")
        self.setMinimumSize(1280, 820)
        self._last_result_image_path = ""
        self._last_result_json_path = ""
        self._last_output_dir = ""
        self._build_ui()
        self._apply_styles()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        title = QLabel("Grape Leaf Disease Detection Workbench")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "Desktop disease recognition system for grape leaves based on YOLO inference and PyQt5 MVC integration."
        )
        subtitle.setObjectName("subtitleLabel")

        header_layout = QVBoxLayout()
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(self._create_project_banner())
        root_layout.addLayout(header_layout)

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self._create_prediction_page(), "Inference Workbench")
        self.main_tabs.addTab(self._create_model_center_page(), "Model Center")
        root_layout.addWidget(self.main_tabs, stretch=1)

        footer = QLabel(
            "Tip: the GUI can use your own trained weights or imported public weights from the local model library."
        )
        footer.setObjectName("footerLabel")
        root_layout.addWidget(footer)

    def _create_prediction_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(14)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(14)
        top_layout.addWidget(self._create_controls_panel(), stretch=1)
        top_layout.addWidget(self._create_status_panel(), stretch=1)
        page_layout.addLayout(top_layout)

        image_layout = QHBoxLayout()
        image_layout.setSpacing(14)
        image_layout.addWidget(self._create_image_card("Original Image", "Select an image to preview."), stretch=1)
        image_layout.addWidget(self._create_image_card("Prediction Preview", "Prediction output will appear here."), stretch=1)
        page_layout.addLayout(image_layout, stretch=1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(14)
        bottom_layout.addWidget(self._create_detections_panel(), stretch=1)
        bottom_layout.addWidget(self._create_log_panel(), stretch=1)
        page_layout.addLayout(bottom_layout, stretch=1)
        return page

    def _create_model_center_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)
        self.center_registered_card = self._create_stat_card("Registered Models", "0")
        self.center_training_card = self._create_stat_card("Training Runs", "0")
        self.center_default_card = self._create_stat_card("Default Model", "--")
        summary_row.addWidget(self.center_registered_card)
        summary_row.addWidget(self.center_training_card)
        summary_row.addWidget(self.center_default_card)
        summary_row.addStretch(1)
        layout.addLayout(summary_row)

        table_row = QHBoxLayout()
        table_row.setSpacing(14)
        table_row.addWidget(self._create_registered_models_panel(), stretch=1)
        table_row.addWidget(self._create_training_runs_panel(), stretch=1)
        layout.addLayout(table_row, stretch=1)

        layout.addWidget(self._create_commands_panel(), stretch=1)
        return page

    def _create_project_banner(self) -> QWidget:
        banner = QFrame()
        banner.setObjectName("bannerFrame")
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(18)

        for title, value, attr_name in (
            ("Task", "Leaf Disease Detection", "banner_task_value"),
            ("Framework", "YOLOv8 / YOLOv5", "banner_framework_value"),
            ("Desktop", "PyQt5 MVC", "banner_ui_value"),
        ):
            card = QFrame()
            card.setObjectName("miniStatCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(4)
            title_label = QLabel(title)
            title_label.setObjectName("miniStatTitle")
            value_label = QLabel(value)
            value_label.setObjectName("miniStatValue")
            setattr(self, attr_name, value_label)
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            layout.addWidget(card)
        layout.addStretch(1)
        return banner

    def _create_controls_panel(self) -> QGroupBox:
        group = QGroupBox("Experiment Control")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setSpacing(10)

        self.image_path_edit = QLineEdit()
        self.weights_path_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.model_alias_edit = QLineEdit()
        self.model_alias_edit.setPlaceholderText("Alias for imported model")
        self.default_model_checkbox = QCheckBox("Set as default")
        self.registered_models_combo = QComboBox()
        self.registered_models_combo.addItem("Select imported model", "")

        self.device_edit = QLineEdit("0")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)

        form.addRow("Image", self._wrap_line_with_button(self.image_path_edit, "Browse", self.open_image_requested.emit))
        form.addRow("Weights", self._wrap_line_with_button(self.weights_path_edit, "Browse", self.open_weights_requested.emit))
        form.addRow("Imported Models", self._wrap_combo_with_buttons())
        form.addRow("Import Alias", self.model_alias_edit)
        form.addRow("", self.default_model_checkbox)
        form.addRow("Output", self._wrap_line_with_button(self.output_dir_edit, "Browse", self.open_output_dir_requested.emit))
        form.addRow("Device", self.device_edit)
        form.addRow("Conf", self.conf_spin)
        form.addRow("IoU", self.iou_spin)
        form.addRow("Image Size", self.imgsz_spin)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.predict_button = QPushButton("Run Prediction")
        self.clear_button = QPushButton("Clear")
        button_row.addWidget(self.predict_button)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

        self.predict_button.clicked.connect(self.predict_requested.emit)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        self.registered_models_combo.currentIndexChanged.connect(
            lambda: self.registered_model_changed.emit(self.registered_models_combo.currentData() or "")
        )
        return group

    def _create_status_panel(self) -> QGroupBox:
        group = QGroupBox("Project Status")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.weights_status_label = QLabel("Weights: not selected")
        self.image_status_label = QLabel("Image: not selected")
        self.result_status_label = QLabel("Result: waiting")
        self.result_status_label.setWordWrap(True)
        self.model_backend_label = QLabel("Backend: not loaded")
        self.output_status_label = QLabel("Output: not generated")

        for label in (
            self.weights_status_label,
            self.image_status_label,
            self.result_status_label,
            self.model_backend_label,
            self.output_status_label,
        ):
            label.setObjectName("statusLabel")
            layout.addWidget(label)

        layout.addWidget(self._create_result_stats_panel())
        layout.addWidget(self._create_result_action_panel())
        layout.addStretch(1)
        return group

    def _create_result_stats_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.stat_detection_count = self._create_stat_card("Detections", "0")
        self.stat_model_type = self._create_stat_card("Backend", "--")
        self.stat_top_class = self._create_stat_card("Top Class", "--")
        layout.addWidget(self.stat_detection_count)
        layout.addWidget(self.stat_model_type)
        layout.addWidget(self.stat_top_class)
        return panel

    def _create_result_action_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.open_result_image_button = QPushButton("Open Result Image")
        self.open_result_json_button = QPushButton("Open Result JSON")
        self.open_output_folder_button = QPushButton("Open Output Folder")
        for button in (
            self.open_result_image_button,
            self.open_result_json_button,
            self.open_output_folder_button,
        ):
            button.setEnabled(False)
            layout.addWidget(button)
        self.open_result_image_button.clicked.connect(lambda: self._open_local_path(self._last_result_image_path))
        self.open_result_json_button.clicked.connect(lambda: self._open_local_path(self._last_result_json_path))
        self.open_output_folder_button.clicked.connect(lambda: self._open_local_path(self._last_output_dir))
        return panel

    def _create_stat_card(self, title: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("statCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("statTitle")
        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        if title == "Detections":
            self.stat_detection_value = value_label
        elif title == "Backend":
            self.stat_backend_value = value_label
        else:
            self.stat_top_class_value = value_label
        return card

    def _create_image_card(self, title: str, placeholder: str) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(320)
        label.setObjectName("imageCanvas")
        label.setWordWrap(True)

        if title == "Original Image":
            self.original_image_label = label
        else:
            self.prediction_image_label = label

        layout.addWidget(label)
        return group

    def _create_detections_panel(self) -> QGroupBox:
        group = QGroupBox("Detection Details")
        layout = QVBoxLayout(group)
        self.detections_table = QTableWidget(0, 6)
        self.detections_table.setHorizontalHeaderLabels(
            ["Class ID", "Class Name", "Confidence", "x1", "y1", "x2, y2"]
        )
        self.detections_table.horizontalHeader().setStretchLastSection(True)
        self.detections_table.verticalHeader().setVisible(False)
        self.detections_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detections_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.detections_table)
        return group

    def _create_registered_models_panel(self) -> QGroupBox:
        group = QGroupBox("Registered Models")
        layout = QVBoxLayout(group)
        self.registered_models_table = QTableWidget(0, 4)
        self.registered_models_table.setHorizontalHeaderLabels(
            ["Alias", "Default", "Weight Path", "Imported At"]
        )
        self.registered_models_table.horizontalHeader().setStretchLastSection(True)
        self.registered_models_table.verticalHeader().setVisible(False)
        self.registered_models_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.registered_models_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.registered_models_table)
        return group

    def _create_training_runs_panel(self) -> QGroupBox:
        group = QGroupBox("Training Results")
        layout = QVBoxLayout(group)
        self.training_runs_table = QTableWidget(0, 7)
        self.training_runs_table.setHorizontalHeaderLabels(
            ["Run Name", "Epochs", "Batch", "Image Size", "mAP50", "mAP50-95", "Best Weight"]
        )
        self.training_runs_table.horizontalHeader().setStretchLastSection(True)
        self.training_runs_table.verticalHeader().setVisible(False)
        self.training_runs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.training_runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.training_runs_table)
        return group

    def _create_commands_panel(self) -> QGroupBox:
        group = QGroupBox("Quick Commands And Paths")
        layout = QVBoxLayout(group)
        self.quick_commands_text = QTextEdit()
        self.quick_commands_text.setReadOnly(True)
        layout.addWidget(self.quick_commands_text)

        button_row = QHBoxLayout()
        self.open_models_dir_button = QPushButton("Open Models Dir")
        self.open_logs_dir_button = QPushButton("Open Logs Dir")
        self.open_reports_dir_button = QPushButton("Open Reports Dir")
        for button in (
            self.open_models_dir_button,
            self.open_logs_dir_button,
            self.open_reports_dir_button,
        ):
            button_row.addWidget(button)
        self.open_models_dir_button.clicked.connect(lambda: self._open_local_path(self._models_dir_path))
        self.open_logs_dir_button.clicked.connect(lambda: self._open_local_path(self._logs_dir_path))
        self.open_reports_dir_button.clicked.connect(lambda: self._open_local_path(self._reports_dir_path))
        layout.addLayout(button_row)
        return group

    def _create_log_panel(self) -> QGroupBox:
        group = QGroupBox("System Logs")
        layout = QVBoxLayout(group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return group

    @staticmethod
    def _wrap_line_with_button(line_edit: QLineEdit, button_text: str, callback) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        button = QPushButton(button_text)
        button.clicked.connect(callback)
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)
        return wrapper

    def _wrap_combo_with_buttons(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        import_button = QPushButton("Import")
        refresh_button = QPushButton("Refresh")
        import_button.clicked.connect(self.import_weights_requested.emit)
        refresh_button.clicked.connect(self.refresh_models_requested.emit)
        layout.addWidget(self.registered_models_combo, stretch=1)
        layout.addWidget(import_button)
        layout.addWidget(refresh_button)
        return wrapper

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f3efe6;
                color: #223126;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #ccd4c6;
                border-radius: 14px;
                margin-top: 10px;
                padding-top: 14px;
                background: #fcfaf5;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }
            QPushButton {
                background: #2f6040;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #274f35;
            }
            QPushButton:disabled {
                background: #8da090;
                color: #eef2ec;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget, QComboBox {
                background: white;
                border: 1px solid #c7d0c2;
                border-radius: 8px;
                padding: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #ccd4c6;
                border-radius: 14px;
                background: #fcfaf5;
                top: -1px;
            }
            QTabBar::tab {
                background: #e7ecdf;
                color: #36523b;
                border: 1px solid #ccd4c6;
                padding: 10px 16px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #fcfaf5;
                color: #23352a;
            }
            QLabel#imageCanvas {
                border: 1px dashed #9fb39d;
                border-radius: 12px;
                background: #f8fbf3;
                color: #5b6b5c;
            }
            QLabel#subtitleLabel {
                color: #526453;
            }
            QLabel#statusLabel {
                background: #eef3e7;
                border-radius: 10px;
                padding: 10px 12px;
            }
            QFrame#bannerFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e2ead8, stop:1 #f4e9d7);
                border: 1px solid #ced7c7;
                border-radius: 14px;
            }
            QFrame#miniStatCard, QFrame#statCard {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid #d0d8cb;
                border-radius: 12px;
            }
            QLabel#miniStatTitle, QLabel#statTitle {
                color: #60725f;
                font-size: 11px;
            }
            QLabel#miniStatValue {
                color: #20352a;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#statValue {
                color: #274734;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#footerLabel {
                color: #617162;
                padding-left: 2px;
            }
            """
        )

    def choose_image_file(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        return file_path

    def choose_weights_file(self) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Weights",
            "",
            "PyTorch Weights (*.pt)",
        )
        return file_path

    def choose_output_directory(self) -> str:
        return QFileDialog.getExistingDirectory(self, "Select Output Directory")

    def set_form_values(
        self,
        *,
        image_path: str = "",
        weights_path: str = "",
        output_dir: str = "",
        device: str = "0",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        self.image_path_edit.setText(image_path)
        self.weights_path_edit.setText(weights_path)
        self.output_dir_edit.setText(output_dir)
        self.device_edit.setText(device)
        self.conf_spin.setValue(conf)
        self.iou_spin.setValue(iou)
        self.imgsz_spin.setValue(imgsz)
        self.weights_status_label.setText(f"Weights: {weights_path or 'not selected'}")
        self.image_status_label.setText(f"Image: {image_path or 'not selected'}")
        self.model_backend_label.setText("Backend: waiting")
        self.output_status_label.setText(f"Output: {output_dir or 'not generated'}")
        self._sync_combo_selection(weights_path)

    def read_form_values(self) -> dict[str, object]:
        return {
            "image_path": self.image_path_edit.text().strip(),
            "weights_path": self.weights_path_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "device": self.device_edit.text().strip() or "0",
            "conf_threshold": float(self.conf_spin.value()),
            "iou_threshold": float(self.iou_spin.value()),
            "image_size": int(self.imgsz_spin.value()),
            "model_alias": self.model_alias_edit.text().strip(),
            "set_default_model": self.default_model_checkbox.isChecked(),
        }

    def update_model_library(self, models: list[dict[str, object]], selected_weights_path: str = "") -> None:
        current_data = selected_weights_path or self.weights_path_edit.text().strip()
        self.registered_models_combo.blockSignals(True)
        self.registered_models_combo.clear()
        self.registered_models_combo.addItem("Select imported model", "")
        for item in models:
            alias = str(item["alias"])
            weights_path = str(item["weights_path"])
            is_default = bool(item.get("is_default", False))
            label = f"{alias} (default)" if is_default else alias
            self.registered_models_combo.addItem(label, weights_path)
        self._sync_combo_selection(current_data)
        self.registered_models_combo.blockSignals(False)

    def _sync_combo_selection(self, weights_path: str) -> None:
        if not hasattr(self, "registered_models_combo"):
            return
        index = self.registered_models_combo.findData(weights_path)
        self.registered_models_combo.setCurrentIndex(index if index >= 0 else 0)

    def set_busy(self, busy: bool) -> None:
        self.predict_button.setDisabled(busy)
        self.clear_button.setDisabled(busy)
        self.result_status_label.setText("Result: running..." if busy else "Result: ready")

    def append_log(self, message: str) -> None:
        self.log_text.append(message)

    def set_result_summary(self, message: str) -> None:
        self.result_status_label.setText(f"Result: {message}")

    def set_runtime_summary(
        self,
        *,
        backend: str,
        detection_count: int,
        top_class: str,
        output_path: str,
    ) -> None:
        self.model_backend_label.setText(f"Backend: {backend}")
        self.output_status_label.setText(f"Output: {output_path}")
        self.stat_detection_value.setText(str(detection_count))
        self.stat_backend_value.setText(backend)
        self.stat_top_class_value.setText(top_class or "--")

    def set_detection_table(self, detections: list[dict[str, object]]) -> None:
        self.detections_table.setRowCount(len(detections))
        for row, detection in enumerate(detections):
            bbox = detection["bbox_xyxy"]
            values = [
                str(detection["class_id"]),
                str(detection["class_name"]),
                f"{float(detection['confidence']):.4f}",
                f"{float(bbox[0]):.1f}",
                f"{float(bbox[1]):.1f}",
                f"{float(bbox[2]):.1f}, {float(bbox[3]):.1f}",
            ]
            for column, value in enumerate(values):
                self.detections_table.setItem(row, column, QTableWidgetItem(value))

    def clear_outputs(self) -> None:
        self.detections_table.setRowCount(0)
        self.log_text.clear()
        self.original_image_label.setText("Select an image to preview.")
        self.original_image_label.setPixmap(QPixmap())
        self.prediction_image_label.setText("Prediction output will appear here.")
        self.prediction_image_label.setPixmap(QPixmap())
        self.result_status_label.setText("Result: waiting")
        self.model_backend_label.setText("Backend: not loaded")
        self.output_status_label.setText("Output: not generated")
        self.stat_detection_value.setText("0")
        self.stat_backend_value.setText("--")
        self.stat_top_class_value.setText("--")
        self.model_alias_edit.clear()
        self.default_model_checkbox.setChecked(False)
        self._last_result_image_path = ""
        self._last_result_json_path = ""
        self._last_output_dir = ""
        self._models_dir_path = ""
        self._logs_dir_path = ""
        self._reports_dir_path = ""
        for button in (
            self.open_result_image_button,
            self.open_result_json_button,
            self.open_output_folder_button,
        ):
            button.setEnabled(False)

    def update_model_center(self, snapshot: dict[str, object]) -> None:
        registered_models = list(snapshot.get("registered_models", []))
        training_runs = list(snapshot.get("training_runs", []))
        paths = dict(snapshot.get("paths", {}))
        quick_commands = list(snapshot.get("quick_commands", []))

        self._models_dir_path = str(paths.get("models_dir", ""))
        self._logs_dir_path = str(paths.get("logs_dir", ""))
        self._reports_dir_path = str(paths.get("reports_dir", ""))

        self.center_registered_card.findChild(QLabel, "statValue")
        self.center_training_card.findChild(QLabel, "statValue")
        self.center_default_card.findChild(QLabel, "statValue")
        self.stat_card_update(self.center_registered_card, str(len(registered_models)))
        self.stat_card_update(self.center_training_card, str(len(training_runs)))
        default_model = next((item for item in registered_models if bool(item.get("is_default", False))), None)
        self.stat_card_update(self.center_default_card, str(default_model["alias"]) if default_model else "--")

        self.registered_models_table.setRowCount(len(registered_models))
        for row, item in enumerate(registered_models):
            values = [
                str(item.get("alias", "")),
                "Yes" if bool(item.get("is_default", False)) else "No",
                str(item.get("weights_path", "")),
                str(item.get("imported_at", "")),
            ]
            for column, value in enumerate(values):
                self.registered_models_table.setItem(row, column, QTableWidgetItem(value))

        self.training_runs_table.setRowCount(len(training_runs))
        for row, item in enumerate(training_runs):
            values = [
                str(item.get("run_name", "")),
                str(item.get("epochs", "")),
                str(item.get("batch", "")),
                str(item.get("imgsz", "")),
                self._format_metric(item.get("map50")),
                self._format_metric(item.get("map5095")),
                str(item.get("best_weight", "")),
            ]
            for column, value in enumerate(values):
                self.training_runs_table.setItem(row, column, QTableWidgetItem(value))

        command_lines = ["Suggested commands:", ""]
        command_lines.extend(quick_commands)
        command_lines.extend(
            [
                "",
                "Important output paths:",
                f"Models: {self._models_dir_path}",
                f"Logs: {self._logs_dir_path}",
                f"Reports: {self._reports_dir_path}",
            ]
        )
        self.quick_commands_text.setPlainText("\n".join(command_lines))

    @staticmethod
    def stat_card_update(card: QFrame, value: str) -> None:
        labels = card.findChildren(QLabel)
        if len(labels) >= 2:
            labels[1].setText(value)

    @staticmethod
    def _format_metric(value) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "--"

    def show_original_image(self, image_path: str) -> None:
        self._set_image_preview(self.original_image_label, image_path)

    def show_prediction_image(self, image_path: str) -> None:
        self._set_image_preview(self.prediction_image_label, image_path)

    def _set_image_preview(self, target: QLabel, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            target.setText(f"Unable to load image:\n{image_path}")
            return
        scaled = pixmap.scaled(
            target.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        target.setPixmap(scaled)
        target.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        image_path = self.image_path_edit.text().strip()
        if image_path and Path(image_path).exists():
            self.show_original_image(image_path)
        prediction_path = getattr(self, "_last_result_image_path", "")
        if prediction_path and Path(prediction_path).exists():
            self.show_prediction_image(prediction_path)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def set_result_artifacts(self, result_image_path: str, result_json_path: str) -> None:
        self._last_result_image_path = result_image_path
        self._last_result_json_path = result_json_path
        self._last_output_dir = str(Path(result_json_path).parent) if result_json_path else ""
        self.open_result_image_button.setEnabled(bool(result_image_path))
        self.open_result_json_button.setEnabled(bool(result_json_path))
        self.open_output_folder_button.setEnabled(bool(self._last_output_dir))

    @staticmethod
    def _open_local_path(path: str) -> None:
        if not path:
            return
        local_path = Path(path)
        target = local_path if local_path.exists() else local_path.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
