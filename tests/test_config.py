from grape_disease_net.config import load_config


def test_load_config_has_required_sections() -> None:
    config = load_config()
    for key in ("project", "paths", "dataset", "training", "inference", "ui"):
        assert key in config
