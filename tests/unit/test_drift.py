"""Unit tests for drift.py: sensor drift, health checks, metrics, Prometheus, SPC."""

from __future__ import annotations

import random

import pytest

from adaptivevision.drift import (
    NOMINAL,
    SENSOR_DRIFT_ALERT,
    DriftDetector,
    HealthCheck,
    MetricsRegistry,
    control_chart,
    ks_two_sample,
    render_metrics,
)

# -----------------------------------------------------------------------------
# Sensor/illumination drift detection
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Health checks, metrics registry, SPC
# -----------------------------------------------------------------------------

def test_counter_increment_and_read() -> None:
    registry = MetricsRegistry()
    registry.increment("inspections")
    registry.increment("inspections", 4)
    assert registry.counter("inspections") == 5


def test_gauge_set_and_read() -> None:
    registry = MetricsRegistry()
    assert registry.gauge("temperature") is None
    registry.set_gauge("temperature", 42.5)
    assert registry.gauge("temperature") == 42.5


def test_histogram_statistics() -> None:
    registry = MetricsRegistry()
    for value in (1.0, 2.0, 3.0, 4.0):
        registry.observe("cycle_time", value)
    histogram = registry.histogram("cycle_time")
    assert histogram.count == 4
    assert histogram.min == 1.0
    assert histogram.max == 4.0
    assert histogram.mean == 2.5
    assert histogram.stddev == pytest.approx(1.118, abs=1e-3)


def test_empty_histogram() -> None:
    registry = MetricsRegistry()
    histogram = registry.histogram("missing")
    assert histogram.count == 0
    assert histogram.mean == 0.0


def test_snapshot_shape() -> None:
    registry = MetricsRegistry()
    registry.increment("pass")
    registry.set_gauge("temp", 1.0)
    registry.observe("cycle", 2.0)
    snapshot = registry.snapshot()
    assert snapshot["counters"] == {"pass": 1}
    assert snapshot["gauges"] == {"temp": 1.0}
    assert snapshot["histograms"]["cycle"]["count"] == 1


def test_control_chart_detects_out_of_control() -> None:
    samples = [10.0] * 19 + [30.0]
    chart = control_chart(samples)
    assert chart.mean == pytest.approx(11.0, abs=0.01)
    assert chart.out_of_control == (19,)


def test_control_chart_empty() -> None:
    chart = control_chart([])
    assert chart.mean == 0.0
    assert chart.out_of_control == ()


def test_control_chart_single_sample() -> None:
    chart = control_chart([5.0])
    assert chart.mean == 5.0
    assert chart.stddev == 0.0
    assert chart.out_of_control == ()


def test_health_check_all_healthy() -> None:
    health = HealthCheck()
    health.register("camera", lambda: True)
    health.register("plc", lambda: True)
    assert health.is_healthy()
    statuses = health.check()
    assert all(status.healthy for status in statuses)


def test_health_check_reports_unhealthy() -> None:
    health = HealthCheck()
    health.register("camera", lambda: True)
    health.register("plc", lambda: False)
    assert not health.is_healthy()
    statuses = {status.name: status.healthy for status in health.check()}
    assert statuses == {"camera": True, "plc": False}


# -----------------------------------------------------------------------------
# Prometheus text exposition
# -----------------------------------------------------------------------------

def test_render_counter() -> None:
    registry = MetricsRegistry()
    registry.increment("inspections", 3)
    payload = render_metrics(registry)
    assert "# TYPE inspections counter" in payload
    assert "inspections_total 3" in payload


def test_render_gauge() -> None:
    registry = MetricsRegistry()
    registry.set_gauge("temperature", 42.5)
    payload = render_metrics(registry)
    assert "# TYPE temperature gauge" in payload
    assert "temperature 42.5" in payload


def test_render_histogram() -> None:
    registry = MetricsRegistry()
    for value in (1.0, 3.0):
        registry.observe("cycle_time", value)
    payload = render_metrics(registry)
    assert "# TYPE cycle_time summary" in payload
    assert "cycle_time_count 2" in payload
    assert "cycle_time_sum 4" in payload
    assert "cycle_time_mean 2" in payload
    assert "cycle_time_min 1" in payload
    assert "cycle_time_max 3" in payload


def test_render_empty_registry() -> None:
    payload = render_metrics(MetricsRegistry())
    assert payload == ""


def test_render_sanitizes_metric_names() -> None:
    registry = MetricsRegistry()
    registry.increment("pass count")
    payload = render_metrics(registry)
    assert "# TYPE passcount counter" in payload
    assert "passcount_total 1" in payload
    assert "pass_count_total" not in payload


def test_render_integer_gauge_without_decimal() -> None:
    registry = MetricsRegistry()

    registry.set_gauge("temperature", 42.0)
    payload = render_metrics(registry)
    assert "temperature 42" in payload
