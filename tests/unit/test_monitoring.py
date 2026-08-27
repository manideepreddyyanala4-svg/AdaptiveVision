"""Unit tests for the M14 monitoring package (metrics, SPC, health)."""

from __future__ import annotations

import pytest

from adaptivevision.monitoring import HealthCheck, MetricsRegistry, control_chart


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
