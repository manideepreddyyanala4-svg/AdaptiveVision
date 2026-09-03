"""Prometheus text exposition format (Milestone M18).

The station exposes runtime telemetry to Prometheus for edge observability. This
module renders a :class:`~adaptivevision.monitoring.metrics.MetricsRegistry`
snapshot into the Prometheus text exposition format (v0.0.4) so a Prometheus
scraper can collect counters, gauges, and histogram summaries directly from the
HTTP API.

The renderer is deliberately dependency-free: it emits plain text lines with
``# HELP`` / ``# TYPE`` metadata and ``_total`` / ``_bucket`` / ``_sum`` /
``_count`` conventions so the metrics are first-class Prometheus series.
"""

from __future__ import annotations

from typing import cast

from adaptivevision.monitoring.metrics import MetricsRegistry

#: Prometheus metric name characters allowed beyond ``[a-zA-Z0-9_]``.
_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


def _sanitize(name: str) -> str:
    """Return ``name`` with characters invalid in a Prometheus metric name removed.

    Args:
        name: The raw metric name.

    Returns:
        A Prometheus-safe metric name.
    """
    return "".join(ch for ch in name if ch in _SAFE)


def _format_value(value: float) -> str:
    """Format a numeric value for the exposition format.

    Args:
        value: The numeric value to format.

    Returns:
        A string representation Prometheus can parse.
    """
    if value == int(value):
        return str(int(value))
    return repr(value)


def render_metrics(registry: MetricsRegistry) -> str:
    """Render a metrics registry snapshot in Prometheus text exposition format.

    Args:
        registry: The metrics registry to render.

    Returns:
        A Prometheus text exposition payload (without a trailing newline).
    """
    snapshot = registry.snapshot()
    lines: list[str] = []

    counters = cast(dict[str, int], snapshot["counters"])
    gauges = cast(dict[str, float], snapshot["gauges"])
    histograms = cast(dict[str, dict[str, float]], snapshot["histograms"])

    for name, count in sorted(counters.items()):
        metric = _sanitize(name)
        lines.append(f"# HELP {metric} Counter {name}")
        lines.append(f"# TYPE {metric} counter")
        lines.append(f"{metric}_total {count}")

    for name, gauge in sorted(gauges.items()):
        metric = _sanitize(name)
        lines.append(f"# HELP {metric} Gauge {name}")
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric} {_format_value(gauge)}")

    for name, histogram in sorted(histograms.items()):
        metric = _sanitize(name)
        lines.append(f"# HELP {metric} Histogram {name}")
        lines.append(f"# TYPE {metric} summary")
        lines.append(f"{metric}_count {int(histogram['count'])}")
        lines.append(
            f"{metric}_sum {_format_value(histogram['mean'] * histogram['count'])}"
        )
        lines.append(f"{metric}_min {_format_value(histogram['min'])}")
        lines.append(f"{metric}_max {_format_value(histogram['max'])}")
        lines.append(f"{metric}_mean {_format_value(histogram['mean'])}")
        lines.append(f"{metric}_stddev {_format_value(histogram['stddev'])}")

    return "\n".join(lines)
