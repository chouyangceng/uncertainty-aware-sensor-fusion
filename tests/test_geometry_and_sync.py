import numpy as np

from uncertainty_sensor_fusion.geometry.transforms import transform_points
from uncertainty_sensor_fusion.synchronization.buffer import TimeSynchronizer


def test_identity_transform_preserves_points():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(transform_points(points, np.eye(4)), points)


def test_synchronizer_rejects_out_of_order_messages():
    sync = TimeSynchronizer(max_delay=0.1)
    sync.push(1.0, "imu", 1)
    assert sync.push(0.5, "imu", 2) is False
