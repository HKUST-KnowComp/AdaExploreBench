from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from viewer.validate_public_data import validate


PUBLIC_DATA = Path(__file__).resolve().parents[2] / "viewer" / "public" / "data"


def test_official_overview_covers_eight_models_without_trajectories():
    manifest = json.loads(gzip.decompress((PUBLIC_DATA / "manifest.json.gz").read_bytes()))
    assert manifest["publication"] == {"mode": "overview", "model_count": 8}
    assert validate(PUBLIC_DATA, compressed=True) == (256, 0)
    assert sorted(path.name for path in PUBLIC_DATA.iterdir()) == ["manifest.json.gz"]


def test_overview_rejects_incomplete_model_results(tmp_path):
    manifest = json.loads(gzip.decompress((PUBLIC_DATA / "manifest.json.gz").read_bytes()))
    manifest["runs"][0]["views"]["all"]["branches"]["flip_50"]["episode_count"] = 1
    (tmp_path / "manifest.json.gz").write_bytes(gzip.compress(json.dumps(manifest).encode()))
    with pytest.raises(ValueError, match="lacks complete default results"):
        validate(tmp_path, compressed=True)
