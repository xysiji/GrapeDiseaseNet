import json
from pathlib import Path


def test_public_weights_catalog_has_recommended_entry() -> None:
    catalog_path = Path("configs/public_weights_catalog.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert any(item["recommended"] for item in catalog)
    assert any(item["id"] == "thesab_grape_leaf_detect_v01" for item in catalog)
