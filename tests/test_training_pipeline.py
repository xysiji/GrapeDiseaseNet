from pathlib import Path

from grape_disease_net.training.pipeline import (
    build_eval_kwargs,
    build_train_kwargs,
    serialize_metrics,
)


class DummyBoxMetrics:
    map = 0.42
    map50 = 0.71
    map75 = 0.38
    mp = 0.68
    mr = 0.6


class DummyMetrics:
    fitness = 0.51
    box = DummyBoxMetrics()
    results_dict = {"metrics/mAP50(B)": 0.71, "metrics/mAP50-95(B)": 0.42}


def test_build_train_kwargs_uses_processed_dataset(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "dataset.yaml").write_text("path: test\n", encoding="utf-8")
    config = {
        "paths": {"processed_detection_dir": str(processed_dir)},
        "dataset": {"random_seed": 42},
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "image_size": 640,
            "device": "cpu",
            "workers": 2,
            "project_dir": str(tmp_path / "logs"),
            "run_name": "exp",
            "patience": 20,
            "cache": False,
            "pretrained": True,
            "optimizer": "auto",
            "amp": False,
            "close_mosaic": 10,
            "degrees": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "scale": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
    }

    kwargs = build_train_kwargs(config, epochs=5, batch=4, run_name="demo")

    assert kwargs["epochs"] == 5
    assert kwargs["batch"] == 4
    assert kwargs["name"] == "demo"
    assert kwargs["data"].endswith("dataset.yaml")


def test_build_eval_kwargs_uses_requested_split(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "dataset.yaml").write_text("path: test\n", encoding="utf-8")
    config = {
        "paths": {"processed_detection_dir": str(processed_dir)},
        "training": {
            "batch_size": 8,
            "image_size": 640,
            "device": "cpu",
            "workers": 2,
        },
    }

    kwargs = build_eval_kwargs(config, split="test")

    assert kwargs["split"] == "test"
    assert kwargs["data"].endswith("dataset.yaml")


def test_serialize_metrics_extracts_box_metrics() -> None:
    payload = serialize_metrics(DummyMetrics())

    assert payload["fitness"] == 0.51
    assert payload["box"]["map50"] == 0.71
    assert payload["results_dict"]["metrics/mAP50(B)"] == 0.71
