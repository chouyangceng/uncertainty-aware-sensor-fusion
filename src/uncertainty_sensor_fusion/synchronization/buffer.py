from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimedMessage:
    timestamp: float
    stream: str
    value: Any


class TimeSynchronizer:
    def __init__(self, max_delay: float = 0.05) -> None:
        if max_delay <= 0:
            raise ValueError("max_delay must be positive")
        self.max_delay = max_delay
        self._last_timestamp: dict[str, float] = {}
        self._buffers: dict[str, list[TimedMessage]] = defaultdict(list)

    def push(self, timestamp: float, stream: str, value: Any) -> bool:
        if timestamp < 0 or not stream or timestamp < self._last_timestamp.get(stream, float("-inf")):
            return False
        self._last_timestamp[stream] = timestamp
        self._buffers[stream].append(TimedMessage(timestamp, stream, value))
        cutoff = timestamp - self.max_delay
        self._buffers[stream] = [message for message in self._buffers[stream] if message.timestamp >= cutoff]
        return True

    def nearest(self, timestamp: float, stream: str) -> TimedMessage | None:
        messages = self._buffers.get(stream, [])
        if not messages:
            return None
        candidate = min(messages, key=lambda message: abs(message.timestamp - timestamp))
        return candidate if abs(candidate.timestamp - timestamp) <= self.max_delay else None
