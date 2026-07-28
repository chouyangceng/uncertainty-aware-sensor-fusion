"""ROS 2 message codecs for the sensor-fusion core.

The module deliberately imports ROS messages lazily.  This keeps the research
algorithms and their normal pytest suite usable on machines without ROS 2.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from uncertainty_sensor_fusion.fusion.occupancy import OccupancyGrid2D
from uncertainty_sensor_fusion.reliability.manager import HealthStatus
from uncertainty_sensor_fusion.tracking.manager import ManagedTrack


@dataclass(frozen=True)
class MessageTypes:
    """ROS message classes used by the bridge.

    Tests can inject small compatible classes, which makes conversion logic
    testable without installing ROS 2.
    """

    occupancy_grid: type
    marker: type
    marker_array: type
    diagnostic_status: type
    diagnostic_array: type


def load_message_types() -> MessageTypes | None:
    """Load standard ROS 2 message classes, or return ``None`` if unavailable."""

    try:
        import rclpy  # noqa: F401  (availability check)
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
        from nav_msgs.msg import OccupancyGrid
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError:
        return None
    return MessageTypes(OccupancyGrid, Marker, MarkerArray, DiagnosticStatus, DiagnosticArray)


def ros2_available() -> bool:
    """Return whether rclpy and all standard output message packages exist."""

    return load_message_types() is not None


def _types_or_load(message_types: MessageTypes | None) -> MessageTypes:
    resolved = message_types or load_message_types()
    if resolved is None:
        raise RuntimeError("ROS 2 is unavailable; install rclpy and standard message packages")
    return resolved


def _set_header(message: Any, frame_id: str, stamp: Any | None) -> None:
    message.header.frame_id = frame_id
    if stamp is not None:
        message.header.stamp.sec = int(stamp.sec)
        message.header.stamp.nanosec = int(getattr(stamp, "nanosec", 0))


def occupancy_grid_to_msg(
    grid: OccupancyGrid2D,
    *,
    frame_id: str = "base_link",
    stamp: Any | None = None,
    message_types: MessageTypes | None = None,
) -> Any:
    """Convert a log-odds grid to ``nav_msgs/OccupancyGrid``.

    Cells with probability 0.5 are emitted as ``-1`` (unknown), following the
    ROS OccupancyGrid convention; measured cells use the inclusive 0..100 scale.
    """

    types = _types_or_load(message_types)
    message = types.occupancy_grid()
    _set_header(message, frame_id, stamp)
    message.info.width = int(grid.width)
    message.info.height = int(grid.height)
    message.info.resolution = float(grid.resolution)
    message.info.origin.position.x = -0.5 * grid.width * grid.resolution
    message.info.origin.position.y = -0.5 * grid.height * grid.resolution
    # OccupancyGrid origin is a Pose; explicitly publish the identity
    # quaternion instead of relying on the middleware's default initialization.
    message.info.origin.orientation.x = 0.0
    message.info.origin.orientation.y = 0.0
    message.info.origin.orientation.z = 0.0
    message.info.origin.orientation.w = 1.0
    probabilities = grid.to_numpy().reshape(-1)
    message.data = [
        (-1 if abs(float(value) - 0.5) < 1e-12 else int(np.rint(value * 100)))
        for value in probabilities
    ]
    return message


def tracks_to_marker_array(
    tracks: Iterable[ManagedTrack],
    *,
    deleted_track_ids: Iterable[int] = (),
    frame_id: str = "base_link",
    stamp: Any | None = None,
    message_types: MessageTypes | None = None,
) -> Any:
    """Convert tracked objects and stale IDs to a ROS marker update."""

    types = _types_or_load(message_types)
    message = types.marker_array()
    active_ids: set[int] = set()
    for track in tracks:
        marker = types.marker()
        _set_header(marker, frame_id, stamp)
        marker.ns = "tracked_objects"
        marker.id = int(track.track_id)
        active_ids.add(marker.id)
        marker.type = getattr(types.marker, "SPHERE", 2)
        marker.action = getattr(types.marker, "ADD", 0)
        marker.pose.position.x = float(track.x)
        marker.pose.position.y = float(track.y)
        marker.pose.position.z = 0.0
        # Markers are axis-aligned spheres, so use a valid identity quaternion.
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.8
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.15, 0.65, 1.0, 0.85
        marker.text = f"id={track.track_id} missed={track.missed}"
        message.markers.append(marker)
    for track_id in sorted(set(map(int, deleted_track_ids)) - active_ids):
        marker = types.marker()
        _set_header(marker, frame_id, stamp)
        marker.ns = "tracked_objects"
        marker.id = track_id
        marker.action = getattr(types.marker, "DELETE", 2)
        message.markers.append(marker)
    return message


def health_to_diagnostics(
    report: dict[str, HealthStatus],
    *,
    frame_id: str = "base_link",
    stamp: Any | None = None,
    message_types: MessageTypes | None = None,
) -> Any:
    """Convert sensor health scores to a ``diagnostic_msgs/DiagnosticArray``."""

    types = _types_or_load(message_types)
    message = types.diagnostic_array()
    _set_header(message, frame_id, stamp)
    for sensor, status in sorted(report.items()):
        item = types.diagnostic_status()
        item.name = f"sensor/{sensor}"
        if status.healthy:
            item.level = (
                getattr(types.diagnostic_status, "OK", 0)
                if status.score >= 0.8
                else getattr(types.diagnostic_status, "WARN", 1)
            )
        else:
            item.level = getattr(types.diagnostic_status, "ERROR", 2)
        item.message = f"score={status.score:.3f} healthy={status.healthy}"
        message.status.append(item)
    return message
