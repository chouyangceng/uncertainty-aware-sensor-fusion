from __future__ import annotations

import numpy as np


class OnlineExtrinsicCalibrator:
    """Estimate a small translation correction from paired point coordinates."""

    def __init__(self, learning_rate: float = 0.2) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = learning_rate
        self.translation = np.zeros(3, dtype=float)

    def update(self, source_points: np.ndarray, target_points: np.ndarray) -> float:
        source = np.asarray(source_points, dtype=float)
        target = np.asarray(target_points, dtype=float)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source and target points must have equal shape (N, 3)")
        residual = target - source - self.translation
        self.translation += self.learning_rate * residual.mean(axis=0)
        return float(np.sqrt(np.mean((target - source - self.translation) ** 2)))

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return points + self.translation
