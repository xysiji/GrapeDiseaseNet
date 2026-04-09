from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from grape_disease_net.common.paths import CONFIG_DIR


DEFAULT_CONFIG_PATH = CONFIG_DIR / "project.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

