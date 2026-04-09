from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from grape_disease_net.common.paths import MODELS_DIR, REPORTS_DIR, ROOT_DIR, ensure_runtime_directories


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def get_dataset_yaml_path(config: dict[str, Any]) -> Path:
    dataset_yaml = resolve_project_path(config["paths"]["processed_detection_dir"]) / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}. Run `python scripts/prepare_dataset.py --overwrite` first."
        )
    return dataset_yaml


def first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_train_kwargs(
    config: dict[str, Any],
    *,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    run_name: str | None = None,
    fraction: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]

    return {
        "data": str(get_dataset_yaml_path(config)),
        "epochs": int(first_defined(epochs, training_cfg["epochs"])),
        "batch": int(first_defined(batch, training_cfg["batch_size"])),
        "imgsz": int(first_defined(imgsz, training_cfg["image_size"])),
        "device": first_defined(device, training_cfg["device"]),
        "workers": int(first_defined(workers, training_cfg["workers"])),
        "project": str(resolve_project_path(training_cfg["project_dir"])),
        "name": first_defined(run_name, training_cfg["run_name"]),
        "patience": int(training_cfg.get("patience", 20)),
        "cache": bool(training_cfg.get("cache", False)),
        "pretrained": bool(training_cfg.get("pretrained", True)),
        "optimizer": training_cfg.get("optimizer", "auto"),
        "seed": int(dataset_cfg["random_seed"]),
        "exist_ok": True,
        "resume": resume,
        "verbose": True,
        "save": True,
        "plots": True,
        "amp": bool(training_cfg.get("amp", True)),
        "close_mosaic": int(training_cfg.get("close_mosaic", 10)),
        "degrees": float(training_cfg.get("degrees", 0.0)),
        "fliplr": float(training_cfg.get("fliplr", 0.5)),
        "flipud": float(training_cfg.get("flipud", 0.0)),
        "hsv_h": float(training_cfg.get("hsv_h", 0.015)),
        "hsv_s": float(training_cfg.get("hsv_s", 0.7)),
        "hsv_v": float(training_cfg.get("hsv_v", 0.4)),
        "scale": float(training_cfg.get("scale", 0.5)),
        "mosaic": float(training_cfg.get("mosaic", 1.0)),
        "mixup": float(training_cfg.get("mixup", 0.0)),
        "copy_paste": float(training_cfg.get("copy_paste", 0.0)),
        "fraction": float(first_defined(fraction, training_cfg.get("fraction", 1.0))),
    }


def build_eval_kwargs(
    config: dict[str, Any],
    *,
    split: str,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    training_cfg = config["training"]
    return {
        "data": str(get_dataset_yaml_path(config)),
        "split": split,
        "batch": int(first_defined(batch, training_cfg["batch_size"])),
        "imgsz": int(first_defined(imgsz, training_cfg["image_size"])),
        "device": first_defined(device, training_cfg["device"]),
        "workers": int(training_cfg["workers"]),
        "project": str(resolve_project_path(training_cfg.get("project_dir", "artifacts/logs"))),
        "name": first_defined(name, f"eval_{split}"),
        "exist_ok": True,
        "plots": True,
        "save_json": False,
        "verbose": True,
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def serialize_metrics(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        payload["results_dict"] = {
            str(key): _safe_float(value) if _safe_float(value) is not None else value
            for key, value in results_dict.items()
        }

    payload["fitness"] = _safe_float(getattr(metrics, "fitness", None))

    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None:
        payload["box"] = {
            "map": _safe_float(getattr(box_metrics, "map", None)),
            "map50": _safe_float(getattr(box_metrics, "map50", None)),
            "map75": _safe_float(getattr(box_metrics, "map75", None)),
            "mp": _safe_float(getattr(box_metrics, "mp", None)),
            "mr": _safe_float(getattr(box_metrics, "mr", None)),
        }
    return payload


def resolve_weights_paths(save_dir: Path) -> dict[str, str]:
    weights_dir = save_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    return {
        "best": str(best_path.resolve()) if best_path.exists() else "",
        "last": str(last_path.resolve()) if last_path.exists() else "",
    }


def archive_weights(run_name: str, weights: dict[str, str]) -> dict[str, str]:
    archived = {"best": "", "last": ""}
    for label in ("best", "last"):
        source = weights.get(label)
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            continue
        target_path = MODELS_DIR / f"{run_name}_{label}.pt"
        shutil.copy2(source_path, target_path)
        archived[label] = str(target_path.resolve())
    return archived


def write_training_report(
    *,
    run_name: str,
    train_kwargs: dict[str, Any],
    save_dir: Path,
    weights: dict[str, str],
    archived_weights: dict[str, str],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
) -> Path:
    ensure_runtime_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{run_name}_{timestamp}_training_report.json"
    payload = {
        "run_name": run_name,
        "save_dir": str(save_dir.resolve()),
        "weights": weights,
        "archived_weights": archived_weights,
        "train_kwargs": train_kwargs,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return report_path


def train_yolov8(
    config: dict[str, Any],
    *,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    run_name: str | None = None,
    fraction: float | None = None,
    resume: bool = False,
    evaluate_test: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    ensure_runtime_directories()
    train_kwargs = build_train_kwargs(
        config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        run_name=run_name,
        fraction=fraction,
        resume=resume,
    )
    model_name = config["training"]["model_name"]

    if dry_run:
        return {
            "dry_run": True,
            "model_name": model_name,
            "train_kwargs": train_kwargs,
        }

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ultralytics not installed. Run `pip install ultralytics` and install torch if needed."
        ) from exc

    model = YOLO(model_name)
    model.train(**train_kwargs)

    save_dir = Path(train_kwargs["project"]) / str(train_kwargs["name"])
    weights = resolve_weights_paths(save_dir)
    archived_weights = archive_weights(str(train_kwargs["name"]), weights)
    best_weights = weights["best"] or model_name
    best_model = YOLO(best_weights)

    validation_metrics = serialize_metrics(
        best_model.val(
            **build_eval_kwargs(
                config,
                split="val",
                batch=batch,
                imgsz=imgsz,
                device=device,
                name=f"{train_kwargs['name']}_val_eval",
            )
        )
    )

    test_metrics = None
    if evaluate_test:
        test_metrics = serialize_metrics(
            best_model.val(
                **build_eval_kwargs(
                    config,
                    split="test",
                    batch=batch,
                    imgsz=imgsz,
                    device=device,
                    name=f"{train_kwargs['name']}_test_eval",
                )
            )
        )

    report_path = write_training_report(
        run_name=str(train_kwargs["name"]),
        train_kwargs=train_kwargs,
        save_dir=save_dir,
        weights=weights,
        archived_weights=archived_weights,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    return {
        "dry_run": False,
        "model_name": model_name,
        "train_kwargs": train_kwargs,
        "save_dir": str(save_dir.resolve()),
        "weights": weights,
        "archived_weights": archived_weights,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "report_path": str(report_path.resolve()),
    }


def evaluate_yolov8(
    config: dict[str, Any],
    *,
    weights_path: str,
    split: str = "test",
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ensure_runtime_directories()
    eval_kwargs = build_eval_kwargs(
        config,
        split=split,
        batch=batch,
        imgsz=imgsz,
        device=device,
    )
    weights = str(resolve_project_path(weights_path))

    if dry_run:
        return {
            "dry_run": True,
            "weights_path": weights,
            "eval_kwargs": eval_kwargs,
        }

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ultralytics not installed. Run `pip install ultralytics` and install torch if needed."
        ) from exc

    metrics = YOLO(weights).val(**eval_kwargs)
    payload = {
        "dry_run": False,
        "weights_path": weights,
        "split": split,
        "metrics": serialize_metrics(metrics),
    }
    report_path = REPORTS_DIR / f"eval_{Path(weights).stem}_{split}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    payload["report_path"] = str(report_path.resolve())
    return payload
