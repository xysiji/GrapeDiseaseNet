from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from grape_disease_net.common.paths import MODELS_DIR, ROOT_DIR


REGISTRY_PATH = MODELS_DIR / "registry.json"
LIBRARY_DIR = MODELS_DIR / "library"


@dataclass(slots=True)
class RegisteredModel:
    alias: str
    weights_path: str
    original_path: str
    imported_at: str
    is_default: bool = False


def _safe_alias(alias: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in alias.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("Model alias cannot be empty.")
    return cleaned


def load_registry() -> list[RegisteredModel]:
    if not REGISTRY_PATH.exists():
        return []
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [RegisteredModel(**item) for item in data]


def save_registry(models: list[RegisteredModel]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = [asdict(model) for model in models]
    REGISTRY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_registered_models() -> list[RegisteredModel]:
    models = load_registry()
    return sorted(models, key=lambda item: (not item.is_default, item.alias.lower()))


def register_model(weights_path: str | Path, alias: str, set_default: bool = False) -> RegisteredModel:
    source = Path(weights_path).resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Weights file not found: {source}")
    if source.suffix.lower() != ".pt":
        raise ValueError("Only .pt weight files are supported.")

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    safe_alias = _safe_alias(alias)
    target_path = LIBRARY_DIR / f"{safe_alias}.pt"
    shutil.copy2(source, target_path)

    if set_default:
        for item in registry:
            item.is_default = False

    model = RegisteredModel(
        alias=safe_alias,
        weights_path=str(target_path.resolve()),
        original_path=str(source),
        imported_at=datetime.now().isoformat(timespec="seconds"),
        is_default=set_default,
    )

    replaced = False
    for index, item in enumerate(registry):
        if item.alias == safe_alias:
            registry[index] = model
            replaced = True
            break
    if not replaced:
        registry.append(model)
    save_registry(registry)
    return model


def resolve_model_choice(alias_or_path: str) -> Path:
    candidate = Path(alias_or_path)
    if candidate.exists():
        return candidate.resolve()

    for model in load_registry():
        if model.alias == alias_or_path:
            return Path(model.weights_path).resolve()
    raise FileNotFoundError(f"Unable to resolve model choice: {alias_or_path}")


def get_default_registered_model() -> RegisteredModel | None:
    for model in load_registry():
        if model.is_default:
            return model
    models = list_registered_models()
    return models[0] if models else None
