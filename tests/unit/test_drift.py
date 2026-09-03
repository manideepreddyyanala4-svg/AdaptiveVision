"""Unit tests for the M21 sensor/illumination drift detector."""

from __future__ import annotations

import random

import pytest

from adaptivevision.drift import (
    NOMINAL,
    SENSOR_DRIFT_ALERT,
    DriftDetector,
    ks_two_sample,
)


def test_ks_two_sample_identical_distributions_has_zero_statistic() -> None:
    sample = [0.1, 0.2, 0.3, 0.4, 0.5]
    statistic, p_value = ks_two_sample(sample, list(sample))
    assert statistic == 0.0
    assert p_value == pytest.approx(1.0)


def test_ks_two_sample_completely_separated_distributions_has_max_statistic() -> None:
    low = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    high = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    statistic, p_value = ks_two_sample(low, high)
    assert statistic == pytest.approx(1.0)
    assert p_value < 0.01


def test_ks_two_sample_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ks_two_sample([], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        ks_two_sample([1.0], [])


def test_ks_two_sample_large_separated_samples_saturates_p_value_to_zero() -> None:
    # Large n and complete separation push the scaled statistic past the
    # survival function's x > 10 short-circuit (see _kolmogorov_survival).
    low = [i / 1000.0 for i in range(1000)]
    high = [100.0 + i / 1000.0 for i in range(1000)]
    statistic, p_value = ks_two_sample(low, high)
    assert statistic == 1.0
    assert p_value == 0.0


def test_ks_two_sample_too_few_points_returns_p_value_one() -> None:
    statistic, p_value = ks_two_sample([0.5], [0.1, 0.9])
    assert p_value == 1.0
    assert statistic == 0.0


def test_ks_two_sample_similar_random_distributions_do_not_falsely_flag() -> None:
    rng = random.Random(1234)
    baseline = [rng.gauss(0.2, 0.05) for _ in range(200)]
    window = [rng.gauss(0.2, 0.05) for _ in range(200)]
    _, p_value = ks_two_sample(window, baseline)
    assert p_value > 0.01


def test_driftdetector_rejects_empty_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        DriftDetector([])


def test_driftdetector_rejects_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        DriftDetector([0.1, 0.2], window_size=0)


def test_driftdetector_rejects_invalid_p_value_threshold() -> None:
    with pytest.raises(ValueError, match="p_value_threshold"):
        DriftDetector([0.1, 0.2], p_value_threshold=0.0)
    with pytest.raises(ValueError, match="p_value_threshold"):
        DriftDetector([0.1, 0.2], p_value_threshold=1.5)


def test_driftdetector_check_before_enough_samples_raises() -> None:
    detector = DriftDetector([0.1, 0.2, 0.3], window_size=10)
    with pytest.raises(ValueError, match="at least"):
        detector.check()


def test_driftdetector_record_returns_none_until_minimum_samples() -> None:
    detector = DriftDetector([0.1, 0.2, 0.3], window_size=10)
    assert detector.record(0.15) is None


def test_driftdetector_reports_nominal_for_stable_scores() -> None:
    rng = random.Random(42)
    baseline = [rng.gauss(0.1, 0.02) for _ in range(150)]
    detector = DriftDetector(baseline, window_size=50, p_value_threshold=0.01)

    report = None
    for _ in range(50):
        report = detector.record(rng.gauss(0.1, 0.02))

    assert report is not None
    assert report.status == NOMINAL
    assert report.drifted is False
    assert report.window_size == 50
    assert report.baseline_size == 150


def test_driftdetector_flags_alert_when_scores_shift_up() -> None:
    rng = random.Random(7)
    baseline = [rng.gauss(0.1, 0.02) for _ in range(150)]
    detector = DriftDetector(baseline, window_size=50, p_value_threshold=0.01)

    report = None
    for _ in range(50):
        # A sustained shift far outside the baseline's range -- a fogged lens
        # or dimmed illuminator would show up exactly like this.
        report = detector.record(rng.gauss(0.6, 0.02))

    assert report is not None
    assert report.status == SENSOR_DRIFT_ALERT
    assert report.drifted is True
    assert report.p_value < 0.01


def test_driftdetector_window_is_bounded_and_reset_clears_it() -> None:
    detector = DriftDetector([0.1, 0.2, 0.3], window_size=5)
    for value in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        detector.record(value)
    assert detector.window_size == 5

    detector.reset()
    assert detector.record(0.5) is None


def test_drift_report_to_dict_round_trips_fields() -> None:
    detector = DriftDetector([0.1, 0.2, 0.3, 0.4], window_size=5, p_value_threshold=0.05)
    detector.record(0.1)
    report = detector.record(0.2)
    assert report is not None
    payload = report.to_dict()
    assert payload["status"] in (NOMINAL, SENSOR_DRIFT_ALERT)
    assert payload["window_size"] == 2
    assert payload["baseline_size"] == 4
