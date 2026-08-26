import numpy as np
import pytest

from uncertainty_sensor_fusion.metrics.calibration import (
    calibration_report,
    normalized_estimation_error_squared,
)


def test_nees_matches_manual_quadratic_form():
    errors = np.array([[1.0, 2.0], [2.0, 0.0]])
    covariances = np.array([np.diag([1.0, 4.0]), np.diag([4.0, 1.0])])
    np.testing.assert_allclose(
        normalized_estimation_error_squared(errors, covariances),
        [2.0, 1.0],
    )


def test_calibrated_gaussian_has_expected_coverage_and_nees():
    rng = np.random.default_rng(21)
    errors = rng.multivariate_normal(np.zeros(2), np.array([[1.0, 0.3], [0.3, 0.7]]), 20000)
    covariances = np.repeat(np.array([[[1.0, 0.3], [0.3, 0.7]]]), errors.shape[0], axis=0)
    report = calibration_report(errors, covariances, confidence_levels=[0.5, 0.9, 0.95])
    assert report["mean_nees"] == pytest.approx(2.0, abs=0.06)
    assert report["empirical_coverage"][2] == pytest.approx(0.95, abs=0.01)
    assert report["calibration_error"] < 0.012


def test_underreported_covariance_is_detected_as_overconfident():
    rng = np.random.default_rng(4)
    errors = rng.normal(size=(8000, 2))
    covariances = np.repeat((np.eye(2) * 0.4)[None, :, :], errors.shape[0], axis=0)
    report = calibration_report(errors, covariances, confidence_levels=[0.8, 0.95])
    assert report["mean_nees"] > 4.5
    assert report["empirical_coverage"][1] < 0.8
    assert report["classification"] == "overconfident"


@pytest.mark.parametrize(
    ("errors", "covariances", "message"),
    [
        (np.zeros((2, 3)), np.repeat(np.eye(3)[None], 2, axis=0), "two-dimensional"),
        (np.zeros((2, 2)), np.zeros((2, 2, 2)), "positive definite"),
        (np.array([[np.nan, 0.0]]), np.eye(2)[None], "finite"),
    ],
)
def test_calibration_rejects_unsupported_or_invalid_inputs(errors, covariances, message):
    with pytest.raises(ValueError, match=message):
        normalized_estimation_error_squared(errors, covariances)
