from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class HealthStatus:
    score: float
    healthy: bool


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
