from __future__ import annotations

import os
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
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QUrl

from grape_disease_net.common.paths import DATA_DIR, MODELS_DIR, PREDICTIONS_DIR


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


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
        self.setWindowTitle("葡萄叶片病害检测识别系统")
        self.resize(1480, 960)
        self.setMinimumSize(1180, 760)
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

        title = QLabel("葡萄叶片病害检测识别系统")
        title_font = QFont()
        title_font.setPointSize(30)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "基于 YOLO 模型推理与 PyQt5 MVC 架构构建的桌面端病害检测与识别平台。"
        )
        subtitle.setObjectName("subtitleLabel")

        header_layout = QVBoxLayout()
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(self._create_project_banner())
        root_layout.addLayout(header_layout)

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self._wrap_tab_page(self._create_prediction_page()), "识别检测")
        self.main_tabs.addTab(self._wrap_tab_page(self._create_model_center_page()), "训练结果 / 模型信息")
        root_layout.addWidget(self.main_tabs, stretch=1)

        footer = QLabel(
            "提示：本系统支持加载本地训练权重，也支持使用模型库中的已导入权重进行推理演示。"
        )
        footer.setObjectName("footerLabel")
        root_layout.addWidget(footer)

    def _wrap_tab_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll

    def _create_prediction_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(16)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)
        top_layout.addWidget(self._create_controls_panel(), stretch=1)
        top_layout.addWidget(self._create_status_panel(), stretch=1)
        page_layout.addLayout(top_layout)

        image_layout = QHBoxLayout()
        image_layout.setSpacing(16)
        image_layout.addWidget(self._create_image_card("原始图像", "请选择一张待识别图像进行预览。"), stretch=1)
        image_layout.addWidget(self._create_image_card("识别结果预览", "完成识别后，带检测框的结果图会显示在这里。"), stretch=1)
        page_layout.addLayout(image_layout, stretch=1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(16)
        bottom_layout.addWidget(self._create_detections_panel(), stretch=1)
        bottom_layout.addWidget(self._create_log_panel(), stretch=1)
        page_layout.addLayout(bottom_layout, stretch=1)
        page_layout.addStretch(1)
        return page

    def _create_model_center_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(16)

        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)
        self.center_registered_card = self._create_stat_card("已注册模型", "0")
        self.center_training_card = self._create_stat_card("训练记录", "0")
        self.center_default_card = self._create_stat_card("当前默认模型", "--")
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
        layout.addStretch(1)
        return page

    def _create_project_banner(self) -> QWidget:
        banner = QFrame()
        banner.setObjectName("bannerFrame")
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(18)

        for title, value, attr_name in (
            ("课题方向", "病害检测与识别", "banner_task_value"),
            ("核心模型", "YOLOv8 / YOLOv5", "banner_framework_value"),
            ("系统架构", "PyQt5 MVC", "banner_ui_value"),
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
        group = QGroupBox("参数设置")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(14)

        self.image_path_edit = QLineEdit()
        self.weights_path_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.model_alias_edit = QLineEdit()
        self.model_alias_edit.setPlaceholderText("请输入导入模型的别名")
        self.default_model_checkbox = QCheckBox("设为默认模型")
        self.registered_models_combo = NoWheelComboBox()
        self.registered_models_combo.addItem("请选择已导入模型", "")

        self.device_combo = NoWheelComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItem("自动", "auto")
        self.device_combo.addItem("GPU 0", "0")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.setToolTip("运行设备。GPU 0 表示使用本机独立显卡，CPU 表示仅使用处理器。")

        self.conf_spin = NoWheelDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip("置信度阈值。数值越高，保留下来的检测结果越严格，误检更少，但也可能漏检。")

        self.iou_spin = NoWheelDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setToolTip("IoU 阈值。用于控制重复检测框的合并强度，常用 0.45 到 0.50。")

        self.imgsz_spin = NoWheelSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setToolTip("输入尺寸。图片会先缩放到这个大小再送入模型，常用 640。")

        self.image_path_edit.setToolTip("选择需要识别的单张叶片图像。")
        self.weights_path_edit.setToolTip("选择模型权重文件。优先使用你自己训练得到的 best.pt。")
        self.output_dir_edit.setToolTip("识别结果的保存目录。建议保留默认目录，便于统一管理。")
        self.model_alias_edit.setToolTip("导入模型到模型库时使用的名称。")
        self.registered_models_combo.setToolTip("这里显示已经导入到本地模型库中的权重。")

        self.image_browse_button = QPushButton("浏览")
        self.weights_browse_button = QPushButton("浏览")
        self.output_browse_button = QPushButton("浏览")

        form.addRow("识别图像", self._wrap_line_with_existing_button(self.image_path_edit, self.image_browse_button))
        form.addRow("模型权重", self._wrap_line_with_existing_button(self.weights_path_edit, self.weights_browse_button))
        form.addRow("模型库", self._wrap_combo_with_buttons())
        form.addRow("导入别名", self.model_alias_edit)
        form.addRow("", self.default_model_checkbox)
        form.addRow("输出目录", self._wrap_line_with_existing_button(self.output_dir_edit, self.output_browse_button))
        form.addRow("运行设备", self.device_combo)
        form.addRow("置信度阈值", self.conf_spin)
        form.addRow("IoU 阈值", self.iou_spin)
        form.addRow("输入尺寸", self.imgsz_spin)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        self.predict_button = QPushButton("开始识别")
        self.clear_button = QPushButton("清空内容")
        button_row.addWidget(self.predict_button)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

        self.predict_button.clicked.connect(lambda _checked=False: self.predict_requested.emit())
        self.clear_button.clicked.connect(lambda _checked=False: self.clear_requested.emit())
        self.image_browse_button.clicked.connect(self._emit_open_image_requested)
        self.weights_browse_button.clicked.connect(self._emit_open_weights_requested)
        self.output_browse_button.clicked.connect(self._emit_open_output_dir_requested)
        self.registered_models_combo.currentIndexChanged.connect(
            lambda _index=0: self.registered_model_changed.emit(self.registered_models_combo.currentData() or "")
        )
        return group

    def _create_status_panel(self) -> QGroupBox:
        group = QGroupBox("运行状态")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.weights_status_label = QLabel("模型权重：未选择")
        self.image_status_label = QLabel("识别图像：未选择")
        self.result_status_label = QLabel("识别结果：等待开始")
        self.result_status_label.setWordWrap(True)
        self.model_backend_label = QLabel("推理后端：未加载")
        self.output_status_label = QLabel("输出文件：尚未生成")

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
        layout.setSpacing(12)

        self.stat_detection_count = self._create_stat_card("检测目标数", "0")
        self.stat_model_type = self._create_stat_card("推理后端", "--")
        self.stat_top_class = self._create_stat_card("主要病害类别", "--")
        layout.addWidget(self.stat_detection_count)
        layout.addWidget(self.stat_model_type)
        layout.addWidget(self.stat_top_class)
        return panel

    def _create_result_action_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.open_result_image_button = QPushButton("打开结果图")
        self.open_result_json_button = QPushButton("打开结果 JSON")
        self.open_output_folder_button = QPushButton("打开输出目录")
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
        if title == "检测目标数":
            self.stat_detection_value = value_label
        elif title == "推理后端":
            self.stat_backend_value = value_label
        else:
            self.stat_top_class_value = value_label
        return card

    def _create_image_card(self, title: str, placeholder: str) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(380)
        label.setObjectName("imageCanvas")
        label.setWordWrap(True)

        if title == "原始图像":
            self.original_image_label = label
        else:
            self.prediction_image_label = label

        layout.addWidget(label)
        return group

    def _create_detections_panel(self) -> QGroupBox:
        group = QGroupBox("检测结果明细")
        layout = QVBoxLayout(group)
        self.detections_table = QTableWidget(0, 6)
        self.detections_table.setHorizontalHeaderLabels(
            ["类别编号", "病害名称", "置信度", "左上 x", "左上 y", "右下 x, y"]
        )
        self.detections_table.horizontalHeader().setStretchLastSection(True)
        self.detections_table.verticalHeader().setVisible(False)
        self.detections_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detections_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.detections_table)
        return group

    def _create_registered_models_panel(self) -> QGroupBox:
        group = QGroupBox("已导入模型")
        layout = QVBoxLayout(group)
        self.registered_models_table = QTableWidget(0, 4)
        self.registered_models_table.setHorizontalHeaderLabels(
            ["模型别名", "默认模型", "权重路径", "导入时间"]
        )
        self.registered_models_table.horizontalHeader().setStretchLastSection(True)
        self.registered_models_table.verticalHeader().setVisible(False)
        self.registered_models_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.registered_models_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.registered_models_table)
        return group

    def _create_training_runs_panel(self) -> QGroupBox:
        group = QGroupBox("训练结果概览")
        layout = QVBoxLayout(group)
        self.training_runs_table = QTableWidget(0, 7)
        self.training_runs_table.setHorizontalHeaderLabels(
            ["训练名称", "轮次", "批大小", "输入尺寸", "mAP50", "mAP50-95", "最佳权重"]
        )
        self.training_runs_table.horizontalHeader().setStretchLastSection(True)
        self.training_runs_table.verticalHeader().setVisible(False)
        self.training_runs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.training_runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.training_runs_table)
        return group

    def _create_commands_panel(self) -> QGroupBox:
        group = QGroupBox("常用命令与目录")
        layout = QVBoxLayout(group)
        self.quick_commands_text = QTextEdit()
        self.quick_commands_text.setReadOnly(True)
        layout.addWidget(self.quick_commands_text)

        button_row = QHBoxLayout()
        self.open_models_dir_button = QPushButton("打开模型目录")
        self.open_logs_dir_button = QPushButton("打开日志目录")
        self.open_reports_dir_button = QPushButton("打开报告目录")
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
        group = QGroupBox("系统日志")
        layout = QVBoxLayout(group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return group

    @staticmethod
    def _wrap_line_with_existing_button(line_edit: QLineEdit, button: QPushButton) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)
        return wrapper

    def _emit_open_image_requested(self, _checked: bool = False) -> None:
        self.open_image_requested.emit()

    def _emit_open_weights_requested(self, _checked: bool = False) -> None:
        self.open_weights_requested.emit()

    def _emit_open_output_dir_requested(self, _checked: bool = False) -> None:
        self.open_output_dir_requested.emit()

    def _wrap_combo_with_buttons(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        import_button = QPushButton("导入")
        refresh_button = QPushButton("刷新")
        import_button.clicked.connect(lambda _checked=False: self.import_weights_requested.emit())
        refresh_button.clicked.connect(lambda _checked=False: self.refresh_models_requested.emit())
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
                font-size: 15px;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI";
            }
            QGroupBox {
                border: 1px solid #ccd4c6;
                border-radius: 14px;
                margin-top: 10px;
                padding-top: 14px;
                background: #fcfaf5;
                font-weight: 600;
                font-size: 17px;
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
                padding: 12px 18px;
                font-weight: 600;
                font-size: 16px;
                min-height: 46px;
                min-width: 92px;
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
                padding: 8px 10px;
                selection-background-color: #cfe0c9;
                font-size: 15px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                min-height: 44px;
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
                padding: 12px 20px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 600;
                font-size: 16px;
                min-width: 190px;
                min-height: 40px;
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
                font-size: 16px;
                padding: 20px;
            }
            QLabel#subtitleLabel {
                color: #526453;
                font-size: 16px;
            }
            QLabel#statusLabel {
                background: #eef3e7;
                border-radius: 10px;
                padding: 12px 14px;
                font-size: 15px;
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
                font-size: 13px;
            }
            QLabel#miniStatValue {
                color: #20352a;
                font-size: 19px;
                font-weight: 700;
            }
            QLabel#statValue {
                color: #274734;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#footerLabel {
                color: #617162;
                padding-left: 2px;
                font-size: 15px;
            }
            QHeaderView::section {
                background: #e9eee3;
                color: #2f4735;
                border: none;
                border-bottom: 1px solid #d4dccf;
                padding: 10px 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QTableWidget {
                gridline-color: #e2e8dd;
                alternate-background-color: #f7faf4;
                font-size: 14px;
            }
            """
        )
        self.detections_table.setAlternatingRowColors(True)
        self.registered_models_table.setAlternatingRowColors(True)
        self.training_runs_table.setAlternatingRowColors(True)
        self.detections_table.verticalHeader().setDefaultSectionSize(34)
        self.registered_models_table.verticalHeader().setDefaultSectionSize(34)
        self.training_runs_table.verticalHeader().setDefaultSectionSize(34)

    def choose_image_file(self) -> str:
        start_dir = self._dialog_start_dir(self.image_path_edit.text().strip(), DATA_DIR)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            start_dir,
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        return file_path

    def choose_weights_file(self) -> str:
        start_dir = self._dialog_start_dir(self.weights_path_edit.text().strip(), MODELS_DIR)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择权重文件",
            start_dir,
            "PyTorch 权重 (*.pt)",
        )
        return file_path

    def choose_output_directory(self) -> str:
        start_dir = self._dialog_start_dir(self.output_dir_edit.text().strip(), PREDICTIONS_DIR)
        return QFileDialog.getExistingDirectory(self, "选择输出目录", start_dir)

    @staticmethod
    def _dialog_start_dir(raw_path: str, fallback_dir: Path) -> str:
        candidate = Path(raw_path).expanduser() if raw_path else fallback_dir
        if candidate.exists():
            return str(candidate if candidate.is_dir() else candidate.parent)
        parent = candidate.parent
        if parent.exists():
            return str(parent)
        return str(fallback_dir)

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
        self._sync_device_selection(device)
        self.conf_spin.setValue(conf)
        self.iou_spin.setValue(iou)
        self.imgsz_spin.setValue(imgsz)
        self.weights_status_label.setText(f"模型权重：{weights_path or '未选择'}")
        self.image_status_label.setText(f"识别图像：{image_path or '未选择'}")
        self.model_backend_label.setText("推理后端：等待推理")
        self.output_status_label.setText(f"输出文件：{output_dir or '尚未生成'}")
        self._sync_combo_selection(weights_path)

    def read_form_values(self) -> dict[str, object]:
        device_value = self.device_combo.currentData()
        if device_value is None:
            device_value = self.device_combo.currentText().strip()
        return {
            "image_path": self.image_path_edit.text().strip(),
            "weights_path": self.weights_path_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "device": str(device_value).strip() or "0",
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
        self.registered_models_combo.addItem("请选择已导入模型", "")
        for item in models:
            alias = str(item["alias"])
            weights_path = str(item["weights_path"])
            is_default = bool(item.get("is_default", False))
            label = f"{alias}（默认）" if is_default else alias
            self.registered_models_combo.addItem(label, weights_path)
        self._sync_combo_selection(current_data)
        self.registered_models_combo.blockSignals(False)

    def _sync_combo_selection(self, weights_path: str) -> None:
        if not hasattr(self, "registered_models_combo"):
            return
        index = self.registered_models_combo.findData(weights_path)
        self.registered_models_combo.setCurrentIndex(index if index >= 0 else 0)

    def _sync_device_selection(self, device: str) -> None:
        if not hasattr(self, "device_combo"):
            return
        index = self.device_combo.findData(device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
            return
        self.device_combo.setEditText(device)

    def set_busy(self, busy: bool) -> None:
        self.predict_button.setDisabled(busy)
        self.clear_button.setDisabled(busy)
        self.result_status_label.setText("识别结果：正在处理中..." if busy else "识别结果：准备就绪")

    def append_log(self, message: str) -> None:
        self.log_text.append(message)

    def set_result_summary(self, message: str) -> None:
        self.result_status_label.setText(f"识别结果：{message}")

    def set_runtime_summary(
        self,
        *,
        backend: str,
        detection_count: int,
        top_class: str,
        output_path: str,
    ) -> None:
        self.model_backend_label.setText(f"推理后端：{backend}")
        self.output_status_label.setText(f"输出文件：{output_path}")
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
        self.original_image_label.setText("请选择一张待识别图像进行预览。")
        self.original_image_label.setPixmap(QPixmap())
        self.prediction_image_label.setText("完成识别后，带检测框的结果图会显示在这里。")
        self.prediction_image_label.setPixmap(QPixmap())
        self.result_status_label.setText("识别结果：等待开始")
        self.model_backend_label.setText("推理后端：未加载")
        self.output_status_label.setText("输出文件：尚未生成")
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
                "是" if bool(item.get("is_default", False)) else "否",
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

        command_lines = ["推荐命令：", ""]
        command_lines.extend(quick_commands)
        command_lines.extend(
            [
                "",
                "重要输出目录：",
                f"模型目录：{self._models_dir_path}",
                f"日志目录：{self._logs_dir_path}",
                f"报告目录：{self._reports_dir_path}",
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
            target.setText(f"图像加载失败：\n{image_path}")
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
        QMessageBox.critical(self, "错误", message)

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
            resolved = str(target.resolve())
            if os.name == "nt":
                os.startfile(resolved)
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(resolved))
