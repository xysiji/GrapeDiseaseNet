from pathlib import Path

import grape_disease_net.common.model_registry as registry_module


def test_register_and_load_model_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(registry_module, "LIBRARY_DIR", tmp_path / "library")

    source = tmp_path / "external.pt"
    source.write_bytes(b"weights")

    model = registry_module.register_model(source, alias="demo_model", set_default=True)
    models = registry_module.list_registered_models()

    assert model.alias == "demo_model"
    assert Path(model.weights_path).exists()
    assert len(models) == 1
    assert models[0].is_default is True
