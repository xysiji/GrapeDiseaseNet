from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.model_registry import register_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an external YOLO weights file into the project.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the external .pt file.")
    parser.add_argument("--alias", type=str, required=True, help="Alias used inside the project model library.")
    parser.add_argument(
        "--default",
        action="store_true",
        help="Mark this imported model as the default project model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = register_model(
        weights_path=args.weights,
        alias=args.alias,
        set_default=args.default,
    )
    print(json.dumps(asdict_like(model), ensure_ascii=False, indent=2))


def asdict_like(model) -> dict[str, object]:
    return {
        "alias": model.alias,
        "weights_path": model.weights_path,
        "original_path": model.original_path,
        "imported_at": model.imported_at,
        "is_default": model.is_default,
    }


if __name__ == "__main__":
    main()
