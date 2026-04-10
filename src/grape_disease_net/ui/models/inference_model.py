from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from grape_disease_net.common.model_registry import (
    get_default_registered_model,
    list_registered_models,
    register_model,
)
from grape_disease_net.common.paths import LOGS_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, ROOT_DIR
from grape_disease_net.config import load_config
from grape_disease_net.inference.predictor import find_default_weights


@dataclass(slots=True)
class UIState:
    image_path: str = ""
    weights_path: str = ""
    output_dir: str = ""
    device: str = "0"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640


class InferenceViewModel:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = str(config_path) if config_path else None
        self.config = load_config(config_path)
        configured_output_dir = str(
            Path(
                self.config.get("inference", {}).get(
                    "default_output_dir",
                    PREDICTIONS_DIR / "gui_output",
                )
            )
        )
        output_dir = Path(configured_output_dir)
        if not output_dir.is_absolute():
            output_dir = (PREDICTIONS_DIR.parents[1] / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.default_output_dir = str(output_dir)
        weights_path = ""
        try:
            weights_path = str(find_default_weights(self.config))
        except FileNotFoundError:
            weights_path = ""

        self.state = UIState(
            weights_path=weights_path,
            output_dir=self.default_output_dir,
            device=str(self.config["training"].get("device", "0")),
            conf_threshold=float(self.config["inference"]["conf_threshold"]),
            iou_threshold=float(self.config["inference"]["iou_threshold"]),
            image_size=int(self.config["training"]["image_size"]),
        )
        self.available_models = list_registered_models()

    def set_image_path(self, image_path: str) -> None:
        self.state.image_path = image_path

    def set_weights_path(self, weights_path: str) -> None:
        self.state.weights_path = weights_path

    def set_output_dir(self, output_dir: str) -> None:
        self.state.output_dir = output_dir

    def set_device(self, device: str) -> None:
        self.state.device = device

    def set_thresholds(self, conf_threshold: float, iou_threshold: float) -> None:
        self.state.conf_threshold = conf_threshold
        self.state.iou_threshold = iou_threshold

    def set_image_size(self, image_size: int) -> None:
        self.state.image_size = image_size

    def refresh_model_library(self) -> list[dict[str, Any]]:
        self.available_models = list_registered_models()
        default_model = get_default_registered_model()
        if default_model and not self.state.weights_path:
            self.state.weights_path = default_model.weights_path
        return [
            {
                "alias": item.alias,
                "weights_path": item.weights_path,
                "is_default": item.is_default,
                "imported_at": item.imported_at,
            }
            for item in self.available_models
        ]

    def import_external_weights(self, weights_path: str, alias: str, set_default: bool = False) -> dict[str, Any]:
        model = register_model(weights_path=weights_path, alias=alias, set_default=set_default)
        self.available_models = list_registered_models()
        self.state.weights_path = model.weights_path
        return {
            "alias": model.alias,
            "weights_path": model.weights_path,
            "is_default": model.is_default,
        }

    def select_registered_model(self, alias_or_path: str) -> str:
        for item in self.available_models:
            if item.alias == alias_or_path or item.weights_path == alias_or_path:
                self.state.weights_path = item.weights_path
                return item.weights_path
        if alias_or_path:
            self.state.weights_path = alias_or_path
        return self.state.weights_path

    def get_model_center_snapshot(self) -> dict[str, Any]:
        return {
            "registered_models": self.refresh_model_library(),
            "training_runs": self._collect_training_runs(),
            "paths": {
                "models_dir": str(MODELS_DIR.resolve()),
                "logs_dir": str(LOGS_DIR.resolve()),
                "reports_dir": str(REPORTS_DIR.resolve()),
            },
            "quick_commands": [
                "python scripts/prepare_dataset.py --overwrite",
                "python scripts/train_yolov8.py --epochs 100 --batch 8 --imgsz 640 --device 0 --workers 2 --run-name grape_yolov8_formal",
                "python scripts/evaluate_yolov8.py --weights artifacts\\models\\grape_yolov8_formal_best.pt --split test --device 0",
                "python scripts/predict_image.py --image <image_path> --weights grape_public_hf_v1 --device 0",
                "python -m grape_disease_net.ui.app",
            ],
        }

    def _collect_training_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for report_path in sorted(REPORTS_DIR.glob("*_training_report.json"), key=lambda path: path.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            box_metrics = payload.get("validation_metrics", {}).get("box", {})
            train_kwargs = payload.get("train_kwargs", {})
            runs.append(
                {
                    "run_name": payload.get("run_name", report_path.stem),
                    "epochs": train_kwargs.get("epochs", ""),
                    "batch": train_kwargs.get("batch", ""),
                    "imgsz": train_kwargs.get("imgsz", ""),
                    "device": train_kwargs.get("device", ""),
                    "map50": box_metrics.get("map50"),
                    "map5095": box_metrics.get("map"),
                    "best_weight": payload.get("archived_weights", {}).get("best")
                    or payload.get("weights", {}).get("best", ""),
                    "report_path": str(report_path.resolve()),
                    "save_dir": payload.get("save_dir", ""),
                }
            )
        return runs

    def predict(self) -> dict[str, Any]:
        if not self.state.image_path:
            raise ValueError("请先选择待识别图像。")
        if not self.state.weights_path:
            raise ValueError("请先选择模型权重文件。")
        command = [
            sys.executable,
            str((ROOT_DIR / "scripts" / "predict_image.py").resolve()),
            "--config",
            str(Path(self.config_path).resolve()) if self.config_path else str((ROOT_DIR / "configs" / "project.yaml").resolve()),
            "--image",
            self.state.image_path,
            "--weights",
            self.state.weights_path,
            "--output-dir",
            self.state.output_dir or self.default_output_dir,
            "--conf",
            str(self.state.conf_threshold),
            "--iou",
            str(self.state.iou_threshold),
            "--imgsz",
            str(self.state.image_size),
            "--device",
            self.state.device,
        ]
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "未知推理错误").strip()
            raise RuntimeError(message)
        return self._extract_prediction_payload(completed.stdout)

    @staticmethod
    def _extract_prediction_payload(stdout: str) -> dict[str, Any]:
        text = stdout.strip()
        if not text:
            raise RuntimeError("推理进程没有返回结果。")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise RuntimeError(f"无法解析推理结果输出：{text}")
