from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HealthStatus:
    score: float
    healthy: bool


@dataclass(frozen=True)
class InnovationStatus:
    nis: float
    healthy: bool
    violation_streak: int
    recovery_streak: int


class NormalizedInnovationMonitor:
    """Detect persistent estimator inconsistency using NIS with hysteresis."""

    def __init__(self, threshold: float, failure_count: int = 3, recovery_count: int = 5) -> None:
        if not np.isfinite(threshold) or threshold <= 0 or failure_count <= 0 or recovery_count <= 0:
            raise ValueError("innovation monitor configuration is invalid")
        self.threshold = float(threshold)
        self.failure_count = int(failure_count)
        self.recovery_count = int(recovery_count)
        self.healthy = True
        self._violations = 0
        self._recoveries = 0

    def update(self, innovation: np.ndarray, covariance: np.ndarray) -> InnovationStatus:
        residual = np.asarray(innovation, dtype=float).reshape(-1)
        matrix = np.asarray(covariance, dtype=float)
        if residual.size == 0 or matrix.shape != (residual.size, residual.size):
            raise ValueError("innovation and covariance dimensions must match")
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(matrix)):
            raise ValueError("innovation statistics must be finite")
        if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
            raise ValueError("covariance must be symmetric positive definite")
        try:
            np.linalg.cholesky(matrix)
            nis = float(residual @ np.linalg.solve(matrix, residual))
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be symmetric positive definite") from exc

        if nis > self.threshold:
            self._violations += 1
            self._recoveries = 0
            if self._violations >= self.failure_count:
                self.healthy = False
        else:
            self._violations = 0
            if not self.healthy:
                self._recoveries += 1
                if self._recoveries >= self.recovery_count:
                    self.healthy = True
                    self._recoveries = 0

        return InnovationStatus(nis, self.healthy, self._violations, self._recoveries)


class SensorHealthManager:
    def __init__(self, sensors: list[str], window: int = 10, threshold: float = 0.5) -> None:
        if not sensors or len(set(sensors)) != len(sensors) or window <= 0 or not 0 <= threshold <= 1:
            raise ValueError("sensor health configuration is invalid")
        self._history = {sensor: deque([True] * window, maxlen=window) for sensor in sensors}
        self.threshold = threshold

    def update(self, sensor: str, valid: bool) -> HealthStatus:
        if sensor not in self._history:
            raise ValueError("unknown sensor")
        self._history[sensor].append(bool(valid))
        return self._status(sensor)

    def _status(self, sensor: str) -> HealthStatus:
        score = sum(self._history[sensor]) / len(self._history[sensor])
        return HealthStatus(float(score), score >= self.threshold)

    def report(self) -> dict[str, HealthStatus]:
        return {sensor: self._status(sensor) for sensor in self._history}
