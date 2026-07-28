"""Optional ROS 2 node wrapping the uncertainty-aware fusion primitives."""

from __future__ import annotations

import numpy as np

from uncertainty_sensor_fusion.fusion.occupancy import OccupancyGrid2D
from uncertainty_sensor_fusion.reliability.manager import SensorHealthManager
from uncertainty_sensor_fusion.tracking.manager import ManagedTrack, MultiObjectTracker

from . import bridge

try:  # Keep importing this module harmless on a normal Python workstation.
    from rclpy.node import Node as _Node
except ImportError:  # pragma: no cover - exercised only outside ROS 2
    _Node = object  # type: ignore[assignment,misc]


def validate_runtime_parameters(
    *,
    point_confidence: float,
    grid_width: int,
    grid_height: int,
    grid_resolution: float,
    association_gate: float,
) -> None:
    """Reject unsafe geometry and probability parameters during startup."""

    if not np.isfinite(point_confidence) or not 0.0 < point_confidence < 1.0:
        raise ValueError("point_confidence must be finite and between 0 and 1")
    if grid_width <= 0:
        raise ValueError("grid_width must be positive")
    if grid_height <= 0:
        raise ValueError("grid_height must be positive")
    if not np.isfinite(grid_resolution) or grid_resolution <= 0.0:
        raise ValueError("grid_resolution must be finite and positive")
    if not np.isfinite(association_gate) or association_gate <= 0.0:
        raise ValueError("association_gate must be finite and positive")


class NearestNeighborAssociator:
    """Assign stable IDs to an unordered point stream using gated distances."""

    def __init__(self, gating_distance: float) -> None:
        if not np.isfinite(gating_distance) or gating_distance <= 0.0:
            raise ValueError("gating_distance must be finite and positive")
        self.gating_distance = float(gating_distance)
        self._next_id = 0

    def associate(
        self, points: np.ndarray, tracks: list[ManagedTrack]
    ) -> list[dict[str, float]]:
        """Greedily pair points and tracks, then allocate monotonic new IDs."""

        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if not np.all(np.isfinite(values)):
            raise ValueError("points must be finite")

        if tracks:
            self._next_id = max(self._next_id, max(track.track_id for track in tracks) + 1)

        candidates: list[tuple[float, int, float, float, int]] = []
        for point_index, point in enumerate(values):
            for track in tracks:
                distance = float(np.hypot(point[0] - track.x, point[1] - track.y))
                if distance <= self.gating_distance:
                    candidates.append(
                        (distance, track.track_id, float(point[0]), float(point[1]), point_index)
                    )
        candidates.sort()

        point_ids: dict[int, int] = {}
        matched_track_ids: set[int] = set()
        for _, track_id, _, _, point_index in candidates:
            if point_index not in point_ids and track_id not in matched_track_ids:
                point_ids[point_index] = track_id
                matched_track_ids.add(track_id)

        unmatched = sorted(
            (index for index in range(len(values)) if index not in point_ids),
            key=lambda index: (float(values[index, 0]), float(values[index, 1]), index),
        )
        for point_index in unmatched:
            point_ids[point_index] = self._next_id
            self._next_id += 1

        return [
            {"id": float(point_ids[index]), "x": float(point[0]), "y": float(point[1])}
            for index, point in enumerate(values)
        ]


class FusionNode(_Node):
    """Point-stream adapter publishing grid, track markers and diagnostics.

    Input ``std_msgs/Float64MultiArray`` data is interpreted as flattened
    ``[x0, y0, x1, y1, ...]`` points in the configured frame.  This compact
    interface is convenient for bags and experiments; a production driver can
    replace only the subscription callback with a PointCloud2 decoder.
    """

    def __init__(self) -> None:
        if _Node is object:
            raise RuntimeError("ROS 2 is unavailable; source /opt/ros/<distro>/setup.bash first")
        super().__init__("uncertainty_sensor_fusion")
        from diagnostic_msgs.msg import DiagnosticArray
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import Float64MultiArray
        from visualization_msgs.msg import MarkerArray

        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("grid_width", 80)
        self.declare_parameter("grid_height", 80)
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("point_confidence", 0.7)
        self.declare_parameter("association_gate", 2.0)
        self.declare_parameter("health_window", 10)
        self.declare_parameter("health_threshold", 0.5)
        self.declare_parameter("max_track_age", 3)
        self.declare_parameter("input_topic", "/fusion/points")
        self.declare_parameter("grid_topic", "/fusion/occupancy_grid")
        self.declare_parameter("tracks_topic", "/fusion/tracks")
        self.declare_parameter("diagnostics_topic", "/fusion/diagnostics")

        get = self.get_parameter
        self._frame_id = str(get("frame_id").value)
        self._confidence = float(get("point_confidence").value)
        grid_width = int(get("grid_width").value)
        grid_height = int(get("grid_height").value)
        grid_resolution = float(get("grid_resolution").value)
        association_gate = float(get("association_gate").value)
        validate_runtime_parameters(
            point_confidence=self._confidence,
            grid_width=grid_width,
            grid_height=grid_height,
            grid_resolution=grid_resolution,
            association_gate=association_gate,
        )
        self._grid = OccupancyGrid2D(
            grid_width,
            grid_height,
            grid_resolution,
        )
        self._tracker = MultiObjectTracker(int(get("max_track_age").value))
        self._associator = NearestNeighborAssociator(association_gate)
        self._association_tracks: list[ManagedTrack] = []
        self._published_track_ids: set[int] = set()
        self._health = SensorHealthManager(
            ["points"], int(get("health_window").value), float(get("health_threshold").value)
        )
        self._grid_pub = self.create_publisher(OccupancyGrid, str(get("grid_topic").value), 10)
        self._tracks_pub = self.create_publisher(MarkerArray, str(get("tracks_topic").value), 10)
        self._diagnostics_pub = self.create_publisher(
            DiagnosticArray, str(get("diagnostics_topic").value), 10
        )
        self._subscription = self.create_subscription(
            Float64MultiArray, str(get("input_topic").value), self._on_points, 10
        )

    def _on_points(self, message: object) -> None:
        values = np.asarray(getattr(message, "data", []), dtype=float)
        valid = values.size > 0 and values.size % 2 == 0 and bool(np.all(np.isfinite(values)))
        timestamp = self.get_clock().now().nanoseconds / 1e9
        if valid:
            points = values.reshape(-1, 2)
            self._grid.update(points, confidence=self._confidence)
            observations = self._associator.associate(points, self._association_tracks)
            tracks = self._tracker.update(observations, timestamp)
        else:
            tracks = self._tracker.update([], timestamp)
        self._association_tracks = tracks
        self._health.update("points", valid)
        stamp = self.get_clock().now().to_msg()
        self._grid_pub.publish(
            bridge.occupancy_grid_to_msg(self._grid, frame_id=self._frame_id, stamp=stamp)
        )
        active_track_ids = {track.track_id for track in tracks}
        deleted_track_ids = self._published_track_ids - active_track_ids
        self._tracks_pub.publish(
            bridge.tracks_to_marker_array(
                tracks,
                deleted_track_ids=deleted_track_ids,
                frame_id=self._frame_id,
                stamp=stamp,
            )
        )
        self._published_track_ids = active_track_ids
        self._diagnostics_pub.publish(
            bridge.health_to_diagnostics(
                self._health.report(), frame_id=self._frame_id, stamp=stamp
            )
        )


def main(args: list[str] | None = None) -> None:
    """Run the node under a ROS 2 executor."""

    if not bridge.ros2_available():
        raise RuntimeError("ROS 2 is unavailable; install rclpy and standard message packages")
    import rclpy

    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
