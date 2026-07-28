from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ManagedTrack:
    track_id: int
    x: float
    y: float
    last_timestamp: float
    missed: int = 0


class MultiObjectTracker:
    def __init__(self, max_age: int = 3) -> None:
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        self.max_age = max_age
        self._tracks: dict[int, ManagedTrack] = {}

    def update(self, observations: list[dict[str, float]], timestamp: float) -> list[ManagedTrack]:
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        observed_ids = set()
        for observation in observations:
            if "id" not in observation or "x" not in observation or "y" not in observation:
                raise ValueError("observation requires id, x and y")
            object_id = int(observation["id"])
            if not np.all(np.isfinite([observation["x"], observation["y"]])):
                raise ValueError("observation coordinates must be finite")
            observed_ids.add(object_id)
            track = self._tracks.get(object_id)
            if track is None:
                track = ManagedTrack(object_id, float(observation["x"]), float(observation["y"]), timestamp)
                self._tracks[object_id] = track
            else:
                track.x, track.y, track.last_timestamp, track.missed = float(observation["x"]), float(observation["y"]), timestamp, 0
        for object_id, track in list(self._tracks.items()):
            if object_id not in observed_ids:
                track.missed += 1
                if track.missed > self.max_age:
                    del self._tracks[object_id]
        return [self._tracks[key] for key in sorted(self._tracks)]
