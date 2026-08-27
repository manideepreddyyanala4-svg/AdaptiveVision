"""Unit tests for the M18 Prometheus text exposition renderer."""

from __future__ import annotations

from adaptivevision.monitoring import MetricsRegistry, render_metrics


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
