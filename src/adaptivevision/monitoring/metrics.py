"""Runtime metrics registry (Milestone M14).

The registry records counters, gauges, and histograms so the station can expose
operational telemetry (cycle times, pass/fail counts, throughput) to the
dashboard and monitoring tooling.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


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
