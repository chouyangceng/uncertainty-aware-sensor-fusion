import numpy as np

from uncertainty_sensor_fusion.calibration import OnlineExtrinsicCalibrator
from uncertainty_sensor_fusion.fusion.occupancy import OccupancyGrid2D
from uncertainty_sensor_fusion.reliability.manager import SensorHealthManager
from uncertainty_sensor_fusion.synchronization.latency import estimate_latency
from uncertainty_sensor_fusion.tracking.manager import MultiObjectTracker

if __name__ == "__main__":
    calibrator = OnlineExtrinsicCalibrator(learning_rate=0.3)
    source = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    for _ in range(5):
        calibrator.update(source, source + np.array([0.2, -0.1, 0.05]))
    grid = OccupancyGrid2D(width=40, height=40, resolution=0.25)
    grid.update(np.array([[1.0, 1.0], [1.5, 1.0], [2.0, 1.0]]), confidence=0.9)
    health = SensorHealthManager(["camera", "lidar", "radar"], window=5)
    health.update("lidar", False)
    tracker = MultiObjectTracker(max_age=3)
    tracks = tracker.update([{"id": 1, "x": 4.0, "y": 1.0}], timestamp=0.0)
    latency = estimate_latency(np.array([0.0, 1.0, 0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, -1.0]), 0.05, 3)
    print({"translation": calibrator.translation.tolist(), "occupied": grid.probability(1.0, 1.0), "health": health.report(), "tracks": tracks, "latency": latency})
