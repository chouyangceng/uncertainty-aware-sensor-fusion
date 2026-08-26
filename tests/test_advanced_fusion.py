import numpy as np


def test_online_calibrator_reduces_translation_residual():
    from uncertainty_sensor_fusion.calibration.online import OnlineExtrinsicCalibrator

    calibrator = OnlineExtrinsicCalibrator(learning_rate=0.5)
    source = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    target = source + np.array([0.2, -0.1, 0.05])
    first = calibrator.update(source, target)
    second = calibrator.update(source, target)
    assert second < first


def test_latency_estimator_recovers_integer_delay():
    from uncertainty_sensor_fusion.synchronization.latency import estimate_latency

    reference = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    observed = np.array([0.0, 0.0, 1.0, 0.0, -1.0])
    assert estimate_latency(reference, observed, dt=0.05, max_lag=3) == 0.05


def test_occupancy_grid_marks_points_and_returns_probability():
    from uncertainty_sensor_fusion.fusion.occupancy import OccupancyGrid2D

    grid = OccupancyGrid2D(width=20, height=20, resolution=0.5)
    grid.update(np.array([[1.0, 1.0], [1.5, 1.0]]), confidence=0.9)
    assert grid.probability(1.0, 1.0) > 0.5
    assert grid.to_numpy().shape == (20, 20)


def test_multi_object_tracker_keeps_track_id():
    from uncertainty_sensor_fusion.tracking.manager import MultiObjectTracker

    tracker = MultiObjectTracker(max_age=2)
    first = tracker.update([{"id": 7, "x": 1.0, "y": 2.0}], timestamp=0.0)
    second = tracker.update([{"id": 7, "x": 1.1, "y": 2.0}], timestamp=0.1)
    assert first[0].track_id == second[0].track_id


def test_sensor_health_manager_degrades_after_invalid_frames():
    from uncertainty_sensor_fusion.reliability.manager import SensorHealthManager

    manager = SensorHealthManager(["camera", "lidar"], window=3)
    manager.update("lidar", False)
    manager.update("lidar", False)
    report = manager.report()
    assert report["lidar"].score < report["camera"].score
    assert report["lidar"].healthy is False


def test_innovation_monitor_uses_hysteresis_for_failure_and_recovery():
    from uncertainty_sensor_fusion.reliability.manager import NormalizedInnovationMonitor

    monitor = NormalizedInnovationMonitor(threshold=5.0, failure_count=2, recovery_count=2)
    covariance = np.eye(2)
    assert monitor.update(np.array([3.0, 0.0]), covariance).healthy
    failed = monitor.update(np.array([3.0, 0.0]), covariance)
    assert failed.nis == 9.0
    assert not failed.healthy
    assert not monitor.update(np.zeros(2), covariance).healthy
    assert monitor.update(np.zeros(2), covariance).healthy


def test_innovation_monitor_rejects_invalid_covariance():
    import pytest

    from uncertainty_sensor_fusion.reliability.manager import NormalizedInnovationMonitor

    monitor = NormalizedInnovationMonitor(threshold=5.0)
    with pytest.raises(ValueError, match="positive definite"):
        monitor.update(np.ones(2), np.array([[1.0, 0.0], [0.0, 0.0]]))
