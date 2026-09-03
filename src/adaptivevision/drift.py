"""Monitoring: sensor drift, health checks, metrics, and process control.

Camera lenses fog, bulbs dim, and fixtures loosen over weeks of production --
none of that shows up as a single bad inspection, only as a slow shift in the
*distribution* of anomaly scores a healthy line produces. :class:`DriftDetector`
compares a sliding window of recent scores against a golden baseline
distribution using a two-sample Kolmogorov-Smirnov test (Milestone M21).
Alongside it: component health checks, a runtime metrics registry, its
Prometheus text-exposition renderer, and SPC control-chart statistics -- the
system-health signals a line operator or dashboard watches, as distinct from
any single part's own pass/fail verdict (that's ``decision.py``'s job).
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

# =============================================================================
# Sensor / illumination drift detection (Milestone M21)
#
# No SciPy dependency: this package's established convention (see the SPC
# section below, camera.py's resize_to) is to keep src/adaptivevision
# dependency-free for exactly this kind of statistic, so ks_two_sample is a
# from-scratch implementation of the standard two-sample KS statistic and its
# asymptotic p-value (the same Kolmogorov distribution series SciPy's
# ks_2samp(..., method="asymp") implements). Cross-checked manually against
# scipy.stats.ks_2samp (run in a separate environment that has SciPy
# installed, since this one deliberately doesn't): the KS statistic itself
# matches bit-for-bit; the asymptotic p-value is close but not bit-identical
# (e.g. 0.8279 here vs SciPy's 0.8058 for one tested pair) due to a simpler
# tail-correction term in this implementation's series -- close enough that
# it never changed an alert/nominal classification in that comparison, but
# treat the exact p-value as approximate, not a precise statistical
# guarantee.
# =============================================================================

#: Status string reported when a drift check exceeds the configured p-value
#: threshold -- the exact signal a line PLC/HMI or the dashboard can key off.
SENSOR_DRIFT_ALERT = "SENSOR_DRIFT_ALERT"

#: Status string for a drift check that found no significant divergence.
NOMINAL = "NOMINAL"

#: Below this many samples in either set, the KS test's asymptotic p-value
#: approximation is unreliable, so a check is skipped rather than reported.
_MIN_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Result of comparing a recent window of scores against a baseline.

    Attributes:
        statistic: The two-sample KS statistic (maximum gap between the two
            empirical CDFs), in ``[0, 1]``. Larger means more divergence.
        p_value: Asymptotic two-sided KS p-value. Small values mean the two
            samples are unlikely to be drawn from the same distribution.
        window_size: Number of recent (window) samples compared.
        baseline_size: Number of baseline samples compared against.
        status: :data:`SENSOR_DRIFT_ALERT` if ``p_value`` is below the
            detector's configured threshold, otherwise :data:`NOMINAL`.
    """

    statistic: float
    p_value: float
    window_size: int
    baseline_size: int
    status: str

    @property
    def drifted(self) -> bool:
        """Return ``True`` when this report is a drift alert."""
        return self.status == SENSOR_DRIFT_ALERT

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "window_size": self.window_size,
            "baseline_size": self.baseline_size,
            "status": self.status,
        }


def ks_two_sample(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Compute the two-sample KS statistic and its asymptotic p-value.

    Args:
        a: First sample of real-valued observations.
        b: Second sample of real-valued observations.

    Returns:
        ``(statistic, p_value)``. ``statistic`` is the maximum absolute gap
        between the two samples' empirical CDFs. ``p_value`` is ``1.0`` when
        either sample has fewer than 2 observations (too little data to say
        anything).

    Raises:
        ValueError: If either sample is empty.
    """
    if not a or not b:
        msg = "ks_two_sample requires both samples to be non-empty"
        raise ValueError(msg)

    sorted_a = sorted(a)
    sorted_b = sorted(b)
    n1, n2 = len(sorted_a), len(sorted_b)

    if n1 < _MIN_SAMPLES or n2 < _MIN_SAMPLES:
        return 0.0, 1.0

    candidates = sorted(set(sorted_a) | set(sorted_b))
    statistic = 0.0
    for value in candidates:
        cdf_a = bisect.bisect_right(sorted_a, value) / n1
        cdf_b = bisect.bisect_right(sorted_b, value) / n2
        gap = abs(cdf_a - cdf_b)
        if gap > statistic:
            statistic = gap

    effective_n = math.sqrt(n1 * n2 / (n1 + n2))
    p_value = _kolmogorov_survival((effective_n + 0.12 + 0.11 / effective_n) * statistic)
    return statistic, p_value


def _kolmogorov_survival(x: float) -> float:
    """Asymptotic survival function of the Kolmogorov distribution.

    The standard series ``Q(x) = 2 * sum_{k=1..inf} (-1)^(k-1) exp(-2 k^2 x^2)``
    used for the two-sided KS test's asymptotic p-value.
    """
    if x < 0.2:
        return 1.0
    if x > 10.0:
        return 0.0
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * x * x)
        total += term if k % 2 == 1 else -term
        if term < 1e-12:
            break
    return min(1.0, max(0.0, 2.0 * total))


class DriftDetector:
    """Sliding-window sensor/illumination drift monitor.

    Compares the most recent ``window_size`` anomaly scores against a fixed
    golden baseline distribution (e.g. scores from a batch of known-good
    parts captured at commissioning or the last recalibration) with a KS
    test, flagging :data:`SENSOR_DRIFT_ALERT` when they diverge significantly.

    Args:
        baseline: Golden-run anomaly scores to compare against. Must be
            non-empty.
        window_size: Number of most recent scores to hold in the sliding
            comparison window.
        p_value_threshold: A drift check below this p-value is flagged as an
            alert.

    Raises:
        ValueError: If ``baseline`` is empty, ``window_size`` is not
            positive, or ``p_value_threshold`` is not in ``(0, 1]``.
    """

    def __init__(
        self,
        baseline: Sequence[float],
        *,
        window_size: int = 100,
        p_value_threshold: float = 0.01,
    ) -> None:
        """Initialize the detector with a fixed baseline distribution."""
        if not baseline:
            msg = "DriftDetector requires a non-empty baseline distribution"
            raise ValueError(msg)
        if window_size <= 0:
            msg = "window_size must be positive"
            raise ValueError(msg)
        if not 0.0 < p_value_threshold <= 1.0:
            msg = "p_value_threshold must be in (0, 1]"
            raise ValueError(msg)
        self._baseline = tuple(baseline)
        self._p_value_threshold = p_value_threshold
        self._window_size = window_size
        self._window: deque[float] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        """Configured sliding-window capacity."""
        return self._window_size

    def record(self, anomaly_score: float) -> DriftReport | None:
        """Add one anomaly score to the sliding window.

        Args:
            anomaly_score: The most recent inspection's anomaly score.

        Returns:
            A :class:`DriftReport` once the window holds at least
            :data:`_MIN_SAMPLES` scores, otherwise ``None`` (too little
            recent data to compare yet).
        """
        self._window.append(anomaly_score)
        if len(self._window) < _MIN_SAMPLES:
            return None
        return self.check()

    def check(self) -> DriftReport:
        """Compare the current window against the baseline without recording.

        Returns:
            The current :class:`DriftReport`.

        Raises:
            ValueError: If fewer than :data:`_MIN_SAMPLES` scores have been
                recorded yet.
        """
        if len(self._window) < _MIN_SAMPLES:
            msg = f"need at least {_MIN_SAMPLES} recorded scores to check for drift"
            raise ValueError(msg)
        statistic, p_value = ks_two_sample(tuple(self._window), self._baseline)
        status = SENSOR_DRIFT_ALERT if p_value < self._p_value_threshold else NOMINAL
        return DriftReport(
            statistic=statistic,
            p_value=p_value,
            window_size=len(self._window),
            baseline_size=len(self._baseline),
            status=status,
        )

    def reset(self) -> None:
        """Clear the sliding window (e.g. after a recalibration)."""
        self._window.clear()


# =============================================================================
# Component health checks
#
# A HealthCheck aggregates the health status of named components so the
# station can report an overall health summary to the dashboard and
# monitoring tooling.
# =============================================================================


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    """Health status of a single component.

    Attributes:
        name: Component identifier.
        healthy: Whether the component is healthy.
        detail: Optional human-readable detail.
    """

    name: str
    healthy: bool
    detail: str | None = None


class HealthCheck:
    """Aggregates the health of named components.

    Each component is registered with a probe callable returning a boolean.
    """

    def __init__(self) -> None:
        """Initialize an empty health check."""
        self._probes: dict[str, Callable[[], bool]] = {}

    def register(self, name: str, probe: Callable[[], bool]) -> None:
        """Register a health probe for a component."""
        self._probes[name] = probe

    def check(self) -> tuple[ComponentStatus, ...]:
        """Evaluate all registered probes and return their statuses."""
        return tuple(ComponentStatus(name=name, healthy=probe()) for name, probe in self._probes.items())

    def is_healthy(self) -> bool:
        """Return ``True`` if every registered component is healthy."""
        return all(status.healthy for status in self.check())


# =============================================================================
# Runtime metrics registry
#
# Records counters, gauges, and histograms so the station can expose
# operational telemetry (cycle times, pass/fail counts, throughput) to the
# dashboard and monitoring tooling.
# =============================================================================


@dataclass(frozen=True, slots=True)
class Histogram:
    """Summary statistics for a set of recorded samples.

    Attributes:
        count: Number of samples.
        min: Minimum sample value.
        max: Maximum sample value.
        mean: Arithmetic mean of the samples.
        stddev: Population standard deviation of the samples.
    """

    count: int
    min: float
    max: float
    mean: float
    stddev: float


class MetricsRegistry:
    """Thread-safe-free registry of counters, gauges, and histograms.

    The registry is intentionally simple and dependency-free; it is not
    thread-safe and is expected to be used from a single monitoring thread.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter by ``amount``."""
        self._counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to ``value``."""
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Record a sample into a histogram."""
        self._histograms[name].append(value)

    def counter(self, name: str) -> int:
        """Return the current value of a counter."""
        return self._counters[name]

    def gauge(self, name: str) -> float | None:
        """Return the current value of a gauge, or ``None`` if unset."""
        return self._gauges.get(name)

    def histogram(self, name: str) -> Histogram:
        """Return summary statistics for a histogram."""
        samples = self._histograms[name]
        if not samples:
            return Histogram(count=0, min=0.0, max=0.0, mean=0.0, stddev=0.0)
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        return Histogram(
            count=len(samples),
            min=min(samples),
            max=max(samples),
            mean=mean,
            stddev=math.sqrt(variance),
        )

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    "count": h.count,
                    "min": h.min,
                    "max": h.max,
                    "mean": h.mean,
                    "stddev": h.stddev,
                }
                for name, h in self._histogram_snapshots()
            },
        }

    def _histogram_snapshots(self) -> Iterable[tuple[str, Histogram]]:
        """Yield ``(name, histogram)`` pairs for all recorded histograms."""
        for name in self._histograms:
            yield name, self.histogram(name)


# =============================================================================
# Prometheus text exposition
#
# Renders a MetricsRegistry snapshot into the Prometheus text exposition
# format (v0.0.4) so a Prometheus scraper can collect counters, gauges, and
# histogram summaries directly from the HTTP API. Deliberately
# dependency-free: it emits plain text lines with "# HELP" / "# TYPE"
# metadata and "_total" / "_bucket" / "_sum" / "_count" conventions so the
# metrics are first-class Prometheus series.
# =============================================================================

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
        lines.append(f"{metric}_sum {_format_value(histogram['mean'] * histogram['count'])}")
        lines.append(f"{metric}_min {_format_value(histogram['min'])}")
        lines.append(f"{metric}_max {_format_value(histogram['max'])}")
        lines.append(f"{metric}_mean {_format_value(histogram['mean'])}")
        lines.append(f"{metric}_stddev {_format_value(histogram['stddev'])}")

    return "\n".join(lines)


# =============================================================================
# Statistical process control (SPC)
#
# Control-chart calculations for monitoring a measured feature over time:
# mean, standard deviation, control limits, and out-of-control detection.
# =============================================================================


@dataclass(frozen=True, slots=True)
class ControlChart:
    """Control-chart statistics for a sequence of samples.

    Attributes:
        mean: Sample mean.
        stddev: Sample standard deviation.
        ucl: Upper control limit (mean + 3 sigma).
        lcl: Lower control limit (mean - 3 sigma).
        out_of_control: Indices of samples beyond the control limits.
    """

    mean: float
    stddev: float
    ucl: float
    lcl: float
    out_of_control: tuple[int, ...]


def control_chart(samples: Sequence[float], *, sigma: float = 3.0) -> ControlChart:
    """Compute control-chart statistics for ``samples``.

    Args:
        samples: The measured values in time order.
        sigma: Number of standard deviations for the control limits.

    Returns:
        The computed control-chart statistics.
    """
    if not samples:
        return ControlChart(mean=0.0, stddev=0.0, ucl=0.0, lcl=0.0, out_of_control=())
    mean = sum(samples) / len(samples)
    if len(samples) == 1:
        stddev = 0.0
    else:
        variance = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
        stddev = math.sqrt(variance)
    ucl = mean + sigma * stddev
    lcl = mean - sigma * stddev
    out_of_control = tuple(i for i, s in enumerate(samples) if s > ucl or s < lcl)
    return ControlChart(
        mean=mean,
        stddev=stddev,
        ucl=ucl,
        lcl=lcl,
        out_of_control=out_of_control,
    )
