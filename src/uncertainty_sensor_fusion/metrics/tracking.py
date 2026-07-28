from __future__ import annotations

import numpy as np


def tracking_summary(errors: np.ndarray, matched: int, ground_truth: int) -> dict[str, float]:
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or not np.all(np.isfinite(errors)) or matched < 0 or ground_truth <= 0:
        raise ValueError("tracking inputs are invalid")
    mota = 1.0 - max(ground_truth - matched, 0) / ground_truth
    return {"mota": float(mota), "motp": float(np.mean(errors) if errors.size else 0.0), "matched": float(matched)}
