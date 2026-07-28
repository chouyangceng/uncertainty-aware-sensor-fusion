from __future__ import annotations

import numpy as np

from ..geometry.transforms import project_points, transform_points


def project_lidar_to_camera(points_lidar: np.ndarray, lidar_to_camera: np.ndarray,
                            intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_points = transform_points(points_lidar, lidar_to_camera)
    return project_points(camera_points, intrinsic)
