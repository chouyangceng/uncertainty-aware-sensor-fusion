"""Optional ROS 2 node wrapping the uncertainty-aware fusion primitives."""

from __future__ import annotations

import numpy as np

from uncertainty_sensor_fusion.fusion.occupancy import OccupancyGrid2D
from uncertainty_sensor_fusion.reliability.manager import SensorHealthManager
from uncertainty_sensor_fusion.tracking.manager import MultiObjectTracker

from . import bridge

try:  # Keep importing this module harmless on a normal Python workstation.
    from rclpy.node import Node as _Node
except ImportError:  # pragma: no cover - exercised only outside ROS 2
    _Node = object  # type: ignore[assignment,misc]


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
        self._grid = OccupancyGrid2D(
            int(get("grid_width").value),
            int(get("grid_height").value),
            float(get("grid_resolution").value),
        )
        self._tracker = MultiObjectTracker(int(get("max_track_age").value))
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
        if valid:
            points = values.reshape(-1, 2)
            self._grid.update(points, confidence=self._confidence)
            timestamp = self.get_clock().now().nanoseconds / 1e9
            observations = [
                {"id": index, "x": point[0], "y": point[1]} for index, point in enumerate(points)
            ]
            tracks = self._tracker.update(observations, timestamp)
        else:
            tracks = self._tracker.update([], self.get_clock().now().nanoseconds / 1e9)
        self._health.update("points", valid)
        stamp = self.get_clock().now().to_msg()
        self._grid_pub.publish(
            bridge.occupancy_grid_to_msg(self._grid, frame_id=self._frame_id, stamp=stamp)
        )
        self._tracks_pub.publish(
            bridge.tracks_to_marker_array(tracks, frame_id=self._frame_id, stamp=stamp)
        )
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
