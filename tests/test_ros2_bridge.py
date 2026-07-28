"""Dependency-free tests for the optional ROS 2 message bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from ros2.sensor_fusion_ros.sensor_fusion_ros import bridge
from uncertainty_sensor_fusion.fusion.occupancy import OccupancyGrid2D
from uncertainty_sensor_fusion.reliability.manager import HealthStatus
from uncertainty_sensor_fusion.tracking.manager import ManagedTrack


@dataclass
class FakeStamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class FakeHeader:
    frame_id: str = ""
    stamp: FakeStamp = field(default_factory=FakeStamp)


@dataclass
class FakeOrigin:
    position: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0))
    orientation: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
    )


@dataclass
class FakeMapInfo:
    width: int = 0
    height: int = 0
    resolution: float = 0.0
    origin: FakeOrigin = field(default_factory=FakeOrigin)


class FakeOccupancyGrid:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.info = FakeMapInfo()
        self.data: list[int] = []


@dataclass
class FakePoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class FakeScale:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class FakeMarker:
    ADD = 0
    DELETE = 2
    SPHERE = 2

    def __init__(self) -> None:
        self.header = FakeHeader()
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = 0
        self.pose = SimpleNamespace(
            position=FakePoint(),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )
        self.scale = FakeScale()
        self.color = SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
        self.text = ""


class FakeMarkerArray:
    def __init__(self) -> None:
        self.markers: list[FakeMarker] = []


class FakeDiagnosticStatus:
    OK = 0
    WARN = 1
    ERROR = 2

    def __init__(self) -> None:
        self.name = ""
        self.level = 0
        self.message = ""
        self.values: list[SimpleNamespace] = []


class FakeDiagnosticArray:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.status: list[FakeDiagnosticStatus] = []


FAKE_TYPES = bridge.MessageTypes(
    occupancy_grid=FakeOccupancyGrid,
    marker=FakeMarker,
    marker_array=FakeMarkerArray,
    diagnostic_status=FakeDiagnosticStatus,
    diagnostic_array=FakeDiagnosticArray,
)


def test_occupancy_grid_conversion_preserves_layout_and_probability() -> None:
    grid = OccupancyGrid2D(width=3, height=2, resolution=0.5)
    grid.update(np.array([[0.0, 0.0]]), confidence=0.9)

    message = bridge.occupancy_grid_to_msg(
        grid, frame_id="map", stamp=FakeStamp(2, 3), message_types=FAKE_TYPES
    )

    assert message.header.frame_id == "map"
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (2, 3)
    assert (message.info.width, message.info.height) == (3, 2)
    assert message.info.resolution == 0.5
    assert message.info.origin.orientation.w == 1.0
    assert len(message.data) == 6
    assert max(message.data) >= 90
    assert min(message.data) == -1  # untouched cells follow ROS's unknown convention


def test_track_conversion_emits_marker_per_track() -> None:
    tracks = [ManagedTrack(7, 1.25, -0.5, last_timestamp=4.0, missed=1)]

    message = bridge.tracks_to_marker_array(
        tracks, frame_id="base_link", stamp=FakeStamp(4), message_types=FAKE_TYPES
    )

    assert len(message.markers) == 1
    marker = message.markers[0]
    assert (marker.id, marker.ns, marker.type) == (7, "tracked_objects", FakeMarker.SPHERE)
    assert (marker.pose.position.x, marker.pose.position.y) == (1.25, -0.5)
    assert marker.pose.orientation.w == 1.0
    assert marker.text == "id=7 missed=1"


def test_track_conversion_deletes_markers_that_disappeared() -> None:
    message = bridge.tracks_to_marker_array(
        [],
        deleted_track_ids={3, 9},
        frame_id="base_link",
        stamp=FakeStamp(5),
        message_types=FAKE_TYPES,
    )

    assert [(marker.id, marker.action) for marker in message.markers] == [
        (3, FakeMarker.DELETE),
        (9, FakeMarker.DELETE),
    ]
    assert all(marker.ns == "tracked_objects" for marker in message.markers)
    assert all(marker.header.stamp.sec == 5 for marker in message.markers)


def test_health_conversion_maps_status_levels_and_scores() -> None:
    report = {
        "camera": HealthStatus(score=0.95, healthy=True),
        "lidar": HealthStatus(score=0.2, healthy=False),
    }

    message = bridge.health_to_diagnostics(report, frame_id="base_link", message_types=FAKE_TYPES)

    assert [item.name for item in message.status] == ["sensor/camera", "sensor/lidar"]
    assert [item.level for item in message.status] == [
        FakeDiagnosticStatus.OK,
        FakeDiagnosticStatus.ERROR,
    ]
    assert "score=0.950" in message.status[0].message
    assert "healthy=False" in message.status[1].message


def test_ros2_runtime_is_optional() -> None:
    assert bridge.ros2_available() in {True, False}
    if not bridge.ros2_available():
        assert bridge.load_message_types() is None
