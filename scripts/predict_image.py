from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.paths import ROOT_DIR, ensure_runtime_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 image prediction entry point.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "project.yaml"),
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to a single image for prediction.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Path to an image directory for batch prediction.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional weights path. Defaults to the latest usable *_best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory for prediction artifacts.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold override.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold override.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, for example cpu or 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max image count for batch prediction.",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="Disable saving annotated visualization images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_directories()
    try:
        from grape_disease_net.config import load_config
        from grape_disease_net.inference.predictor import (
            predict_image_directory,
            predict_single_image,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependencies. Run `pip install -r requirements.txt` first."
        ) from exc

    if not args.image and not args.image_dir:
        raise SystemExit("Please provide `--image` or `--image-dir`.")

    config = load_config(args.config)
    save_visualization = not args.no_save_vis

    if args.image:
        result = predict_single_image(
            config=config,
            image_path=args.image,
            weights_path=args.weights or None,
            output_dir=args.output_dir or None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_visualization=save_visualization,
        )
    else:
        result = predict_image_directory(
            config=config,
            image_dir=args.image_dir,
            weights_path=args.weights or None,
            output_dir=args.output_dir or None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_visualization=save_visualization,
            limit=args.limit,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
