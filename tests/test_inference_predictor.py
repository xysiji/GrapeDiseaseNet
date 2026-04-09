from pathlib import Path

from grape_disease_net.inference.predictor import find_default_weights, first_defined


def test_first_defined_returns_zero_and_false() -> None:
    assert first_defined(None, 0, 1) == 0
    assert first_defined(None, False, True) is False


def test_find_default_weights_prefers_configured_path(tmp_path: Path) -> None:
    weights = tmp_path / "custom_best.pt"
    weights.write_bytes(b"weights")
    config = {
        "inference": {"default_weights": str(weights)},
        "training": {"run_name": "unused"},
    }

    resolved = find_default_weights(config)

    assert resolved == weights.resolve()
