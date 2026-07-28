from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..fusion.geometry_fusion import fuse_observations


def run(seed: int = 3, frames: int = 60, output_dir: str | Path = "artifacts/sensor-fusion") -> dict[str, float | str]:
    if frames < 2:
        raise ValueError("frames must be at least 2")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = np.column_stack([np.linspace(5.0, 35.0, frames), 1.5 * np.sin(np.linspace(0, 2, frames))])
    baseline_errors: list[float] = []
    fused_errors: list[float] = []
    missing_lidar_errors: list[float] = []
    for index, (x, y) in enumerate(truth):
        camera = [{"id": 1, "x": x + rng.normal(0, 0.45), "y": y + rng.normal(0, 0.25)}]
        lidar = None if index > frames // 2 else [{"id": 1, "x": x + rng.normal(0, 0.12), "y": y + rng.normal(0, 0.10)}]
        radar = [{"id": 1, "x": x + rng.normal(0, 0.30), "y": y, "velocity": 0.5}]
        baseline = camera[0]
        fused = fuse_observations(camera, lidar, radar)[0]
        baseline_errors.append(float(np.hypot(baseline["x"] - x, baseline["y"] - y)))
        fused_errors.append(float(np.hypot(fused.x - x, fused.y - y)))
        if lidar is None:
            missing_lidar_errors.append(fused_errors[-1])
    summary: dict[str, float | str] = {
        "baseline": "camera_only",
        "frames": frames,
        "camera_only_position_rmse": float(np.sqrt(np.mean(np.square(baseline_errors)))),
        "fused_position_rmse": float(np.sqrt(np.mean(np.square(fused_errors)))),
        "missing_lidar_position_rmse": float(np.sqrt(np.mean(np.square(missing_lidar_errors)))),
    }
    np.savetxt(
        output / "trace.csv",
        np.column_stack([truth, baseline_errors, fused_errors]),
        delimiter=",",
        header="truth_x,truth_y,camera_error,fused_error",
        comments="",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
