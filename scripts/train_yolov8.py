from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.paths import ROOT_DIR, ensure_runtime_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 training entry point.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "project.yaml"),
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
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
        help="Training device, for example cpu, 0 or 0,1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name under the training project directory.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Dataset fraction used for quick smoke runs, between 0 and 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the previous run with the same run name.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test split evaluation after training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved training config without launching training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_directories()
    try:
        from grape_disease_net.config import load_config
        from grape_disease_net.training.pipeline import train_yolov8
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependencies. Run `pip install -r requirements.txt` first."
        ) from exc

    config = load_config(args.config)
    result = train_yolov8(
        config=config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        run_name=args.run_name,
        fraction=args.fraction,
        resume=args.resume,
        evaluate_test=not args.skip_test,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
