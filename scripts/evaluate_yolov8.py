from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.paths import ROOT_DIR, ensure_runtime_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 evaluation entry point.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "project.yaml"),
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the model weights file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override the batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override the input image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device, for example cpu, 0 or 0,1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved evaluation config without launching evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_directories()
    try:
        from grape_disease_net.config import load_config
        from grape_disease_net.training.pipeline import evaluate_yolov8
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependencies. Run `pip install -r requirements.txt` first."
        ) from exc

    config = load_config(args.config)
    result = evaluate_yolov8(
        config=config,
        weights_path=args.weights,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
