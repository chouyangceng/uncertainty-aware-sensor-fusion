from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uncertainty_calibration import run_study


def test_fast_calibration_study_generates_reproducible_visual_artifacts(tmp_path: Path):
    output = run_study(output_dir=tmp_path, seed=23, fast=True)
    expected = {
        "calibration.json",
        "coverage.csv",
        "calibration_curve.png",
        "nees_distribution.png",
        "whitened_residuals.png",
        "summary.md",
    }
    assert expected == {item.name for item in output.iterdir()}
    raw = (output / "calibration.json").read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    payload = json.loads(raw)
    assert payload["metadata"]["seed"] == 23
    assert payload["scenarios"]["calibrated"]["classification"] == "calibrated"
    assert payload["scenarios"]["underreported_40pct"]["classification"] == "overconfident"
    assert payload["scenarios"]["conservative_200pct"]["classification"] == "underconfident"
    assert "95%" in (output / "summary.md").read_text(encoding="utf-8")
    assert all((output / name).stat().st_size > 1000 for name in expected if name.endswith(".png"))

    with pytest.raises(FileExistsError, match="not empty"):
        run_study(output_dir=tmp_path, seed=23, fast=True)
