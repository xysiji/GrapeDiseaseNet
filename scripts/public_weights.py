from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grape_disease_net.common.model_registry import register_model
from grape_disease_net.common.paths import ROOT_DIR


CATALOG_PATH = ROOT_DIR / "configs" / "public_weights_catalog.json"
PUBLIC_DIR = ROOT_DIR / "artifacts" / "models" / "public"


def load_catalog() -> list[dict[str, object]]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def find_entry(entry_id: str) -> dict[str, object]:
    for item in load_catalog():
        if item["id"] == entry_id:
            return item
    raise KeyError(f"Unknown public weight id: {entry_id}")


def download_file(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target_path)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or download public model weights recommended for this project.")
    parser.add_argument("--list", action="store_true", help="List all known public weight candidates.")
    parser.add_argument("--id", type=str, default="", help="Public weight id from the catalog.")
    parser.add_argument("--download", action="store_true", help="Download the selected weight.")
    parser.add_argument("--register", action="store_true", help="Register the downloaded weight into the local model library.")
    parser.add_argument("--alias", type=str, default="", help="Alias used when registering the weight.")
    parser.add_argument("--default", action="store_true", help="Mark the registered weight as the default model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list or not args.id:
        print(json.dumps(load_catalog(), ensure_ascii=False, indent=2))
        return

    entry = find_entry(args.id)
    result: dict[str, object] = {"entry": entry}
    downloaded_path = None

    if args.download or args.register:
        suffix = Path(str(entry["download_url"]).split("?")[0]).suffix or ".pt"
        downloaded_path = PUBLIC_DIR / f"{entry['id']}{suffix}"
        download_file(str(entry["download_url"]), downloaded_path)
        result["downloaded_path"] = str(downloaded_path.resolve())

    if args.register:
        alias = args.alias.strip() or str(entry["id"])
        if downloaded_path is None:
            raise RuntimeError("Download path missing before registration.")
        registered = register_model(
            weights_path=downloaded_path,
            alias=alias,
            set_default=args.default,
        )
        result["registered"] = {
            "alias": registered.alias,
            "weights_path": registered.weights_path,
            "original_path": registered.original_path,
            "imported_at": registered.imported_at,
            "is_default": registered.is_default,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
