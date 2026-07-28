from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanTrack:
    state: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4))
    gate_threshold: float = 9.21
    age: int = 1
    hits: int = 1

    @classmethod
    def from_measurement(cls, measurement: list[float] | np.ndarray) -> KalmanTrack:
        measurement = np.asarray(measurement, dtype=float)
        if measurement.shape != (2,) or not np.all(np.isfinite(measurement)):
            raise ValueError("measurement must be a finite vector with shape (2,)")
        state = np.array([measurement[0], measurement[1], 0.0, 0.0])
        return cls(state=state)

    def predict(self, dt: float = 0.1) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + np.eye(4) * 0.01
        self.age += 1
        return self.state.copy()

    def update(self, measurement: list[float] | np.ndarray) -> bool:
        measurement = np.asarray(measurement, dtype=float)
        if measurement.shape != (2,) or not np.all(np.isfinite(measurement)):
            raise ValueError("measurement must be a finite vector with shape (2,)")
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + np.eye(2) * 0.04
        distance = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        if distance > self.gate_threshold:
            return False
        gain = np.linalg.solve(innovation_cov, (self.covariance @ observation.T).T).T
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ observation) @ self.covariance
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        self.hits += 1
        return True
