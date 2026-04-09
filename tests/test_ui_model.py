from pathlib import Path

from grape_disease_net.ui.models.inference_model import InferenceViewModel


def test_ui_model_reads_defaults_from_config(tmp_path: Path) -> None:
    weights_path = tmp_path / "demo_best.pt"
    weights_path.write_bytes(b"pt")
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: demo",
                "paths:",
                f"  processed_detection_dir: {tmp_path.as_posix()}",
                "dataset:",
                "  random_seed: 42",
                "training:",
                "  run_name: demo",
                "  image_size: 640",
                "  device: 0",
                "inference:",
                "  conf_threshold: 0.25",
                "  iou_threshold: 0.45",
                f"  default_weights: {weights_path.as_posix()}",
                "  save_visualization: true",
                "ui:",
                "  window_title: demo",
                "  minimum_width: 1200",
                "  minimum_height: 800",
            ]
        ),
        encoding="utf-8",
    )

    model = InferenceViewModel(config_path=config_path)

    assert model.state.weights_path == str(weights_path)
    assert model.state.image_size == 640
    assert model.state.device == "0"
