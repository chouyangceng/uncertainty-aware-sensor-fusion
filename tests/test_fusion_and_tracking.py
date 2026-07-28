def test_missing_lidar_degrades_gracefully():
    from uncertainty_sensor_fusion.fusion.geometry_fusion import fuse_observations

    result = fuse_observations(camera=[{"id": 1, "x": 10.0}], lidar=None, radar=[])
    assert result[0].confidence < 1.0
    assert result[0].source == "camera"


def test_kalman_track_rejects_far_outlier():
    from uncertainty_sensor_fusion.tracking.kalman import KalmanTrack

    track = KalmanTrack.from_measurement([0.0, 0.0])
    assert track.update([100.0, 100.0]) is False
