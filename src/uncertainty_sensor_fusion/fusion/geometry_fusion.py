from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FusedObservation:
    object_id: int
    x: float
    y: float
    velocity: float
    confidence: float
    source: str


def _as_map(observations: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if observations is None:
        return {}
    result = {}
    for item in observations:
        if "id" not in item or "x" not in item:
            raise ValueError("each observation requires id and x")
        result[int(item["id"])] = item
    return result


def fuse_observations(camera: list[dict[str, Any]] | None,
                      lidar: list[dict[str, Any]] | None,
                      radar: list[dict[str, Any]] | None) -> list[FusedObservation]:
    camera_map, lidar_map, radar_map = _as_map(camera), _as_map(lidar), _as_map(radar)
    ids = sorted(set(camera_map) | set(lidar_map) | set(radar_map))
    result: list[FusedObservation] = []
    for object_id in ids:
        items = [mapping[object_id] for mapping in (camera_map, lidar_map, radar_map) if object_id in mapping]
        xs = [float(item["x"]) for item in items]
        ys = [float(item.get("y", 0.0)) for item in items]
        velocities = [float(item.get("velocity", 0.0)) for item in items]
        confidence = min(0.99, 0.45 + 0.18 * len(items))
        source = "+".join(name for name, mapping in (("camera", camera_map), ("lidar", lidar_map), ("radar", radar_map)) if object_id in mapping)
        result.append(FusedObservation(object_id, float(np.mean(xs)), float(np.mean(ys)), float(np.mean(velocities)), confidence, source))
    return result
