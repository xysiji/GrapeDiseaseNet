from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grape_disease_net.common.model_registry import get_default_registered_model, resolve_model_choice
from grape_disease_net.common.paths import MODELS_DIR, PREDICTIONS_DIR, ROOT_DIR, ensure_runtime_directories


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def find_default_weights(config: dict[str, Any]) -> Path:
    inference_cfg = config.get("inference", {})
    configured = inference_cfg.get("default_weights")
    if configured:
        resolved = resolve_project_path(configured)
        if resolved.exists():
            return resolved

    registered = get_default_registered_model()
    if registered is not None:
        resolved = Path(registered.weights_path)
        if resolved.exists():
            return resolved.resolve()

    run_name = config.get("training", {}).get("run_name")
    if run_name:
        candidate = MODELS_DIR / f"{run_name}_best.pt"
        if candidate.exists():
            return candidate.resolve()

    candidates = sorted(MODELS_DIR.glob("*_best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        "No usable weight file found. Pass `--weights` explicitly or train a model first."
    )


def validate_image_path(image_path: str | Path) -> Path:
    resolved = resolve_project_path(image_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Expected an image file path, got directory: {resolved}")
    return resolved


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(slots=True)
class PredictionArtifact:
    source_image: str
    visualized_image: str
    json_path: str
    backend: str
    num_detections: int
    detections: list[Detection]


class GrapeDiseasePredictor:
    def __init__(self, weights_path: str | Path, device: str | None = None) -> None:
        self.weights_path = resolve_project_path(weights_path)
        if not self.weights_path.exists():
            self.weights_path = resolve_model_choice(str(weights_path))
        self.device = device
        self.backend = "ultralytics"
        self.model = self._load_model()

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO

            self.backend = "ultralytics"
            return YOLO(str(self.weights_path))
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "ultralytics not installed. Run `pip install -r requirements.txt` first."
            ) from exc
        except Exception as exc:
            if not self._is_yolov5_weight_error(exc):
                raise
            self.backend = "yolov5_hub"
            return self._load_yolov5_model()

    @staticmethod
    def _is_yolov5_weight_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "yolov5 model" in message or "not forwards compatible with yolov8" in message

    def _load_yolov5_model(self) -> Any:
        import torch

        os.environ.setdefault("YOLOv5_AUTOINSTALL", "false")
        original_posix = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        try:
            return torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(self.weights_path),
                force_reload=False,
                trust_repo=True,
            )
        finally:
            pathlib.PosixPath = original_posix

    @staticmethod
    def _normalize_torch_device(device: str | None) -> str | None:
        value = str(device).strip() if device is not None else ""
        if not value:
            return None
        if value.lower() == "cpu":
            return "cpu"
        if value.isdigit():
            return f"cuda:{value}"
        return value

    def predict_image(
        self,
        image_path: str | Path,
        *,
        save_dir: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
        save_visualization: bool = True,
    ) -> PredictionArtifact:
        image = validate_image_path(image_path)
        output_dir = resolve_project_path(save_dir) if save_dir else PREDICTIONS_DIR / image.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "ultralytics":
            return self._predict_with_ultralytics(
                image=image,
                output_dir=output_dir,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=first_defined(device, self.device),
                save_visualization=save_visualization,
            )
        return self._predict_with_yolov5(
            image=image,
            output_dir=output_dir,
            conf=conf,
            imgsz=imgsz,
            device=first_defined(device, self.device),
            save_visualization=save_visualization,
        )

    def _predict_with_ultralytics(
        self,
        *,
        image: Path,
        output_dir: Path,
        conf: float,
        iou: float,
        imgsz: int,
        device: str | None,
        save_visualization: bool,
    ) -> PredictionArtifact:
        result = self.model.predict(
            source=str(image),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=False,
            verbose=False,
        )[0]

        names = result.names
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            cls_values = boxes.cls.tolist() if boxes.cls is not None else []
            conf_values = boxes.conf.tolist() if boxes.conf is not None else []
            xyxy_values = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            for class_id, confidence, bbox_xyxy in zip(cls_values, conf_values, xyxy_values):
                class_index = int(class_id)
                detections.append(
                    Detection(
                        class_id=class_index,
                        class_name=str(names[class_index]),
                        confidence=float(confidence),
                        bbox_xyxy=[float(value) for value in bbox_xyxy],
                    )
                )

        visualized_path = output_dir / f"{image.stem}_pred.jpg"
        if save_visualization:
            self._save_visualization(result, visualized_path)

        json_path = output_dir / f"{image.stem}_pred.json"
        artifact = PredictionArtifact(
            source_image=str(image.resolve()),
            visualized_image=str(visualized_path.resolve()) if save_visualization else "",
            json_path=str(json_path.resolve()),
            backend=self.backend,
            num_detections=len(detections),
            detections=detections,
        )
        json_path.write_text(
            json.dumps(asdict(artifact), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact

    def _predict_with_yolov5(
        self,
        *,
        image: Path,
        output_dir: Path,
        conf: float,
        imgsz: int,
        device: str | None,
        save_visualization: bool,
    ) -> PredictionArtifact:
        import cv2

        normalized_device = self._normalize_torch_device(device)
        if normalized_device:
            self.model.to(normalized_device)
        self.model.conf = conf

        results = self.model(str(image), size=imgsz)
        rows = results.xyxy[0].tolist()
        names = getattr(results, "names", getattr(self.model, "names", {}))
        detections: list[Detection] = []
        for row in rows:
            x1, y1, x2, y2, score, class_id = row
            index = int(class_id)
            detections.append(
                Detection(
                    class_id=index,
                    class_name=str(names[index]),
                    confidence=float(score),
                    bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                )
            )

        visualized_path = output_dir / f"{image.stem}_pred.jpg"
        if save_visualization:
            rendered = results.render()[0]
            cv2.imwrite(str(visualized_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

        json_path = output_dir / f"{image.stem}_pred.json"
        artifact = PredictionArtifact(
            source_image=str(image.resolve()),
            visualized_image=str(visualized_path.resolve()) if save_visualization else "",
            json_path=str(json_path.resolve()),
            backend=self.backend,
            num_detections=len(detections),
            detections=detections,
        )
        json_path.write_text(
            json.dumps(asdict(artifact), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact

    def predict_directory(
        self,
        image_dir: str | Path,
        *,
        save_dir: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
        save_visualization: bool = True,
        limit: int | None = None,
    ) -> list[PredictionArtifact]:
        source_dir = resolve_project_path(image_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {source_dir}")

        images = sorted(
            path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if limit is not None:
            images = images[:limit]

        batch_dir = resolve_project_path(save_dir) if save_dir else PREDICTIONS_DIR / source_dir.name
        batch_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[PredictionArtifact] = []
        for image in images:
            artifacts.append(
                self.predict_image(
                    image_path=image,
                    save_dir=batch_dir / image.stem,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    save_visualization=save_visualization,
                )
            )
        summary_path = batch_dir / "batch_summary.json"
        summary_path.write_text(
            json.dumps([asdict(item) for item in artifacts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifacts

    @staticmethod
    def _save_visualization(result: Any, target_path: Path) -> None:
        import cv2

        image_bgr = result.plot()
        cv2.imwrite(str(target_path), image_bgr)


def predict_single_image(
    config: dict[str, Any],
    *,
    image_path: str,
    weights_path: str | None = None,
    output_dir: str | None = None,
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    save_visualization: bool | None = None,
) -> dict[str, Any]:
    ensure_runtime_directories()
    inference_cfg = config["inference"]
    predictor = GrapeDiseasePredictor(
        weights_path=weights_path or find_default_weights(config),
        device=device,
    )
    artifact = predictor.predict_image(
        image_path=image_path,
        save_dir=output_dir,
        conf=float(first_defined(conf, inference_cfg["conf_threshold"])),
        iou=float(first_defined(iou, inference_cfg["iou_threshold"])),
        imgsz=int(first_defined(imgsz, config["training"]["image_size"])),
        device=device,
        save_visualization=bool(first_defined(save_visualization, inference_cfg["save_visualization"])),
    )
    return asdict(artifact)


def predict_image_directory(
    config: dict[str, Any],
    *,
    image_dir: str,
    weights_path: str | None = None,
    output_dir: str | None = None,
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    save_visualization: bool | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    ensure_runtime_directories()
    inference_cfg = config["inference"]
    predictor = GrapeDiseasePredictor(
        weights_path=weights_path or find_default_weights(config),
        device=device,
    )
    artifacts = predictor.predict_directory(
        image_dir=image_dir,
        save_dir=output_dir,
        conf=float(first_defined(conf, inference_cfg["conf_threshold"])),
        iou=float(first_defined(iou, inference_cfg["iou_threshold"])),
        imgsz=int(first_defined(imgsz, config["training"]["image_size"])),
        device=device,
        save_visualization=bool(first_defined(save_visualization, inference_cfg["save_visualization"])),
        limit=limit,
    )
    return [asdict(item) for item in artifacts]
