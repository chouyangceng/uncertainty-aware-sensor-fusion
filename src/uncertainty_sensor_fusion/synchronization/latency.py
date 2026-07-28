from __future__ import annotations

import numpy as np


def estimate_latency(reference: np.ndarray, observed: np.ndarray, dt: float, max_lag: int = 20) -> float:
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if reference.ndim != 1 or observed.shape != reference.shape or dt <= 0 or max_lag < 0:
        raise ValueError("signals must be equal 1-D arrays and timing parameters must be valid")
    best_lag, best_error = 0, float("inf")
    for lag in range(-min(max_lag, len(reference) - 1), min(max_lag, len(reference) - 1) + 1):
        if lag >= 0:
            left, right = reference[: len(reference) - lag], observed[lag:]
        else:
            left, right = reference[-lag:], observed[: len(observed) + lag]
        error = float(np.mean((left - right) ** 2))
        if error < best_error:
            best_lag, best_error = lag, error
    return float(best_lag * dt)
