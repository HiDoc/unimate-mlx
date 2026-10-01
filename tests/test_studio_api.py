import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from unimate_mlx.studio_api import _validated_payload


@pytest.fixture
def engine(tmp_path: Path) -> SimpleNamespace:
    model_dir = tmp_path / "preview"
    model_dir.mkdir()
    (model_dir / "model_ema.safetensors").touch()
    (model_dir / "dataset_stats.npy").touch()
    model = SimpleNamespace(
        checkpoint=model_dir / "model_ema.safetensors",
        stats=model_dir / "dataset_stats.npy",
    )
    return SimpleNamespace(_models={"preview": model})


def test_validated_payload_accepts_ui_defaults(engine: SimpleNamespace) -> None:
    prompts, options = _validated_payload(
        json.dumps({"prompts": [{"text": "Walk forward"}], "model": "preview"}), engine
    )

    assert prompts == [{"text": "Walk forward", "label": ""}]
    assert options["mode"] == "fast"
    assert options["blendFrames"] == 8


@pytest.mark.parametrize(
    "override",
    [
        {"model": []},
        {"mode": []},
        {"datasetType": {}},
        {"animMode": None},
        {"guidance": float("nan")},
        {"guidance": float("inf")},
        {"seed": 1.5},
        {"blendFrames": True},
    ],
)
def test_validated_payload_returns_422_for_malformed_values(
    engine: SimpleNamespace, override: dict[str, object]
) -> None:
    payload = {"prompts": [{"text": "Walk forward"}], **override}

    with pytest.raises(HTTPException) as error:
        _validated_payload(json.dumps(payload), engine)

    assert error.value.status_code == 422
