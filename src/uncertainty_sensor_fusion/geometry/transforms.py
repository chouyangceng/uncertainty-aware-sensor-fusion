from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("points must have shape (N, 3) and transform shape (4, 4)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(transform)):
        raise ValueError("points and transform must be finite")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ transform.T
    if np.any(np.abs(transformed[:, 3]) < 1e-12):
        raise ValueError("transform produced points at infinity")
    return transformed[:, :3] / transformed[:, 3:4]


def project_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or intrinsic.shape != (3, 3):
        raise ValueError("camera points must have shape (N, 3) and intrinsic shape (3, 3)")
    valid = points[:, 2] > 1e-6
    uvw = points @ intrinsic.T
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return pixels, valid
