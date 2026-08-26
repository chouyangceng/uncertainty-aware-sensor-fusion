from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def normalized_estimation_error_squared(
    errors: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Compute per-sample 2-D position NEES with strict covariance checks."""

    residuals = np.asarray(errors, dtype=float)
    matrices = np.asarray(covariances, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1:] != (2,):
        raise ValueError("errors must contain two-dimensional position residuals")
    if matrices.shape != (residuals.shape[0], 2, 2):
        raise ValueError("covariances must have shape (samples, 2, 2)")
    if residuals.shape[0] == 0:
        raise ValueError("calibration requires at least one sample")
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(matrices)):
        raise ValueError("errors and covariances must contain only finite values")
    if not np.allclose(matrices, matrices.transpose(0, 2, 1), rtol=1e-7, atol=1e-10):
        raise ValueError("covariances must be symmetric positive definite")
    try:
        np.linalg.cholesky(matrices)
        solved = np.linalg.solve(matrices, residuals[..., None])[..., 0]
    except np.linalg.LinAlgError as error:
        raise ValueError("covariances must be symmetric positive definite") from error
    nees = np.einsum("ni,ni->n", residuals, solved)
    if not np.all(np.isfinite(nees)):
        raise FloatingPointError("NEES became non-finite")
    return nees


def calibration_report(
    errors: np.ndarray,
    covariances: np.ndarray,
    *,
    confidence_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
) -> dict[str, Any]:
    """Compare nominal 2-D confidence ellipses with empirical coverage.

    For two degrees of freedom the chi-square quantile is available in closed
    form: ``q(p) = -2 log(1-p)``. This keeps the core metric dependency-free.
    """

    levels = np.asarray(confidence_levels, dtype=float)
    if (
        levels.ndim != 1
        or levels.size == 0
        or not np.all(np.isfinite(levels))
        or np.any(levels <= 0)
        or np.any(levels >= 1)
        or np.any(np.diff(levels) <= 0)
    ):
        raise ValueError("confidence_levels must be finite, increasing, and between zero and one")
    nees = normalized_estimation_error_squared(errors, covariances)
    thresholds = -2.0 * np.log1p(-levels)
    empirical = np.asarray([np.mean(nees <= threshold) for threshold in thresholds])
    gaps = empirical - levels
    mean_nees = float(np.mean(nees))
    if mean_nees > 2.2:
        classification = "overconfident"
    elif mean_nees < 1.8:
        classification = "underconfident"
    else:
        classification = "calibrated"
    return {
        "dimensions": 2,
        "samples": int(nees.size),
        "expected_mean_nees": 2.0,
        "mean_nees": mean_nees,
        "median_nees": float(np.median(nees)),
        "p95_nees": float(np.percentile(nees, 95)),
        "confidence_levels": levels.tolist(),
        "chi_square_thresholds": thresholds.tolist(),
        "empirical_coverage": empirical.tolist(),
        "coverage_gap": gaps.tolist(),
        "calibration_error": float(np.mean(np.abs(gaps))),
        "maximum_calibration_error": float(np.max(np.abs(gaps))),
        "classification": classification,
    }
