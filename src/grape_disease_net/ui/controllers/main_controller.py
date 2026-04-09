from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from grape_disease_net.ui.models.inference_model import InferenceViewModel
from grape_disease_net.ui.views.main_window import MainWindow


class PredictionWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model: InferenceViewModel) -> None:
        super().__init__()
        self.model = model

    def run(self) -> None:
        try:
            result = self.model.predict()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class MainController:
    def __init__(self, model: InferenceViewModel, view: MainWindow) -> None:
        self.model = model
        self.view = view
        self._thread: QThread | None = None
        self._worker: PredictionWorker | None = None
        self._connect_signals()
        self._populate_defaults()

    def _connect_signals(self) -> None:
        self.view.open_image_requested.connect(self.handle_open_image)
        self.view.open_weights_requested.connect(self.handle_open_weights)
        self.view.open_output_dir_requested.connect(self.handle_open_output_dir)
        self.view.import_weights_requested.connect(self.handle_import_weights)
        self.view.refresh_models_requested.connect(self.handle_refresh_models)
        self.view.registered_model_changed.connect(self.handle_registered_model_changed)
        self.view.predict_requested.connect(self.handle_predict)
        self.view.clear_requested.connect(self.handle_clear)

    def _populate_defaults(self) -> None:
        state = self.model.state
        self.view.set_form_values(
            image_path=state.image_path,
            weights_path=state.weights_path,
            output_dir=state.output_dir,
            device=state.device,
            conf=state.conf_threshold,
            iou=state.iou_threshold,
            imgsz=state.image_size,
        )
        self.view.update_model_library(
            self.model.refresh_model_library(),
            selected_weights_path=state.weights_path,
        )
        self.view.update_model_center(self.model.get_model_center_snapshot())
        if state.weights_path:
            self.view.append_log(f"Default weights loaded: {state.weights_path}")

    def handle_open_image(self) -> None:
        path = self.view.choose_image_file()
        if not path:
            return
        self.model.set_image_path(path)
        self.view.image_path_edit.setText(path)
        self.view.image_status_label.setText(f"Image: {path}")
        self.view.show_original_image(path)
        self.view.append_log(f"Image selected: {path}")

    def handle_open_weights(self) -> None:
        path = self.view.choose_weights_file()
        if not path:
            return
        self.model.set_weights_path(path)
        self.view.weights_path_edit.setText(path)
        self.view.weights_status_label.setText(f"Weights: {path}")
        self.view.append_log(f"Weights selected: {path}")

    def handle_import_weights(self) -> None:
        path = self.view.choose_weights_file()
        if not path:
            return
        values = self.view.read_form_values()
        alias = str(values["model_alias"]).strip() or Path(path).stem
        imported = self.model.import_external_weights(
            weights_path=path,
            alias=alias,
            set_default=bool(values["set_default_model"]),
        )
        self.view.weights_path_edit.setText(imported["weights_path"])
        self.view.weights_status_label.setText(f"Weights: {imported['weights_path']}")
        self.view.update_model_library(
            self.model.refresh_model_library(),
            selected_weights_path=str(imported["weights_path"]),
        )
        self.view.update_model_center(self.model.get_model_center_snapshot())
        self.view.append_log(f"Model imported: alias={imported['alias']}")

    def handle_refresh_models(self) -> None:
        self.view.update_model_library(
            self.model.refresh_model_library(),
            selected_weights_path=self.model.state.weights_path,
        )
        self.view.update_model_center(self.model.get_model_center_snapshot())
        self.view.append_log("Model library refreshed.")

    def handle_registered_model_changed(self, weights_path: str) -> None:
        if not weights_path:
            return
        resolved = self.model.select_registered_model(weights_path)
        self.view.weights_path_edit.setText(resolved)
        self.view.weights_status_label.setText(f"Weights: {resolved}")

    def handle_open_output_dir(self) -> None:
        path = self.view.choose_output_directory()
        if not path:
            return
        self.model.set_output_dir(path)
        self.view.output_dir_edit.setText(path)
        self.view.append_log(f"Output directory selected: {path}")

    def handle_predict(self) -> None:
        values = self.view.read_form_values()
        self.model.set_image_path(str(values["image_path"]))
        self.model.set_weights_path(str(values["weights_path"]))
        self.model.set_output_dir(str(values["output_dir"]))
        self.model.set_device(str(values["device"]))
        self.model.set_thresholds(
            float(values["conf_threshold"]),
            float(values["iou_threshold"]),
        )
        self.model.set_image_size(int(values["image_size"]))

        self.view.set_busy(True)
        self.view.append_log("Prediction started.")

        self._thread = QThread()
        self._worker = PredictionWorker(self.model)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_prediction_success)
        self._worker.failed.connect(self._on_prediction_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_prediction_success(self, result: dict) -> None:
        self.view.set_busy(False)
        detections = result["detections"]
        top_class = str(detections[0]["class_name"]) if detections else "No target"
        self.view.set_result_summary(
            f"{result['num_detections']} detections, result saved."
        )
        self.view.set_runtime_summary(
            backend=str(result.get("backend", "unknown")),
            detection_count=int(result["num_detections"]),
            top_class=top_class,
            output_path=str(result["json_path"]),
        )
        self.view.set_detection_table(detections)
        self.view.show_original_image(result["source_image"])
        if result["visualized_image"]:
            self.view.show_prediction_image(result["visualized_image"])
        self.view.set_result_artifacts(
            result_image_path=str(result["visualized_image"]),
            result_json_path=str(result["json_path"]),
        )
        self.view.append_log(f"Prediction finished: {result['json_path']}")
        self.view.append_log(f"Detections: {result['num_detections']}")
        self.view.append_log(f"Backend: {result.get('backend', 'unknown')}")

    def _on_prediction_failed(self, message: str) -> None:
        self.view.set_busy(False)
        self.view.set_result_summary("prediction failed")
        self.view.append_log(f"Prediction failed: {message}")
        self.view.show_error(message)

    def handle_clear(self) -> None:
        state = self.model.state
        self.model.set_image_path("")
        self.model.set_output_dir("")
        self.view.set_form_values(
            image_path="",
            weights_path=state.weights_path,
            output_dir="",
            device=state.device,
            conf=state.conf_threshold,
            iou=state.iou_threshold,
            imgsz=state.image_size,
        )
        self.view.update_model_library(
            self.model.refresh_model_library(),
            selected_weights_path=state.weights_path,
        )
        self.view.update_model_center(self.model.get_model_center_snapshot())
        self.view.clear_outputs()
