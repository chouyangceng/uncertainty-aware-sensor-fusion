from __future__ import annotations

import numpy as np


class OccupancyGrid2D:
    def __init__(self, width: int, height: int, resolution: float = 0.2, logodds_clip: float = 8.0) -> None:
        if width <= 0 or height <= 0 or resolution <= 0 or logodds_clip <= 0:
            raise ValueError("grid dimensions and resolution must be positive")
        self.width, self.height, self.resolution = width, height, resolution
        self.logodds_clip = logodds_clip
        self._logodds = np.zeros((height, width), dtype=float)

    def _index(self, x: float, y: float) -> tuple[int, int] | None:
        column = int(np.floor(x / self.resolution + self.width / 2))
        row = int(np.floor(y / self.resolution + self.height / 2))
        if 0 <= row < self.height and 0 <= column < self.width:
            return row, column
        return None

    def update(self, points: np.ndarray, confidence: float = 0.7) -> None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not 0 < confidence < 1:
            raise ValueError("points must have shape (N, 2) and confidence must be in (0, 1)")
        increment = float(np.log(confidence / (1.0 - confidence)))
        for x, y in points:
            index = self._index(float(x), float(y))
            if index is not None:
                self._logodds[index] = np.clip(self._logodds[index] + increment, -self.logodds_clip, self.logodds_clip)

    def probability(self, x: float, y: float) -> float:
        index = self._index(x, y)
        if index is None:
            return 0.5
        return float(1.0 / (1.0 + np.exp(-self._logodds[index])))

    def to_numpy(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._logodds))
