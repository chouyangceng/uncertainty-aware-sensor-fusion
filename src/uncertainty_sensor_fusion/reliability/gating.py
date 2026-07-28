from __future__ import annotations

import numpy as np


def confidence_weight(covariance: np.ndarray, floor: float = 1e-6) -> float:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite square matrix")
    uncertainty = float(np.trace(covariance))
    return float(1.0 / (1.0 + max(uncertainty, floor)))


def sensor_health(valid_count: int, total_count: int) -> float:
    if total_count <= 0 or valid_count < 0 or valid_count > total_count:
        raise ValueError("valid_count and total_count are inconsistent")
    return float(valid_count / total_count)
