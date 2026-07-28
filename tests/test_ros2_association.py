"""Dependency-free tests for ROS 2 point-to-track association."""

from __future__ import annotations

import numpy as np
import pytest

from ros2.sensor_fusion_ros.sensor_fusion_ros import fusion_node
from uncertainty_sensor_fusion.tracking.manager import ManagedTrack


def _tracks_from_observations(observations: list[dict[str, float]]) -> list[ManagedTrack]:
    return [
        ManagedTrack(int(item["id"]), float(item["x"]), float(item["y"]), 0.0)
        for item in observations
    ]


def test_association_keeps_ids_when_point_order_changes() -> None:
    associator = fusion_node.NearestNeighborAssociator(gating_distance=1.0)
    first = associator.associate(np.array([[4.0, 0.0], [0.0, 0.0]]), [])
    tracks = _tracks_from_observations(first)

    second = associator.associate(np.array([[0.1, 0.0], [4.1, 0.0]]), tracks)

    ids_by_x = {round(item["x"]): int(item["id"]) for item in second}
    first_ids_by_x = {round(item["x"]): int(item["id"]) for item in first}
    assert ids_by_x == first_ids_by_x


def test_association_assigns_new_id_outside_gate() -> None:
    associator = fusion_node.NearestNeighborAssociator(gating_distance=0.5)
    first = associator.associate(np.array([[0.0, 0.0]]), [])

    second = associator.associate(np.array([[2.0, 0.0]]), _tracks_from_observations(first))

    assert int(second[0]["id"]) != int(first[0]["id"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"point_confidence": 0.0}, "point_confidence"),
        ({"point_confidence": 1.0}, "point_confidence"),
        ({"grid_width": 0}, "grid_width"),
        ({"grid_height": -1}, "grid_height"),
        ({"grid_resolution": 0.0}, "grid_resolution"),
        ({"association_gate": -0.1}, "association_gate"),
    ],
)
def test_runtime_parameter_validation_rejects_invalid_ranges(
    kwargs: dict[str, float | int], message: str
) -> None:
    values: dict[str, float | int] = {
        "point_confidence": 0.7,
        "grid_width": 80,
        "grid_height": 80,
        "grid_resolution": 0.2,
        "association_gate": 2.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        fusion_node.validate_runtime_parameters(**values)
