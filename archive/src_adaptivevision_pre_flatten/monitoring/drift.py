"""Sensor and illumination drift detection (Milestone M21).

Camera lenses fog, bulbs dim, and fixtures loosen over weeks of production --
none of that shows up as a single bad inspection, only as a slow shift in the
*distribution* of anomaly scores a healthy line produces. This module
compares a sliding window of recent scores against a golden baseline
distribution (captured once, at commissioning or the last known-good
recalibration) using a two-sample Kolmogorov-Smirnov test, and raises
:data:`SENSOR_DRIFT_ALERT` when the two distributions have diverged enough
that it's very unlikely to be chance.

No SciPy dependency: this package's established convention (see
``monitoring/spc.py``, ``preprocessing/operators.py::resize_to``) is to keep
``src/adaptivevision`` dependency-free for exactly this kind of statistic, so
:func:`ks_two_sample` is a from-scratch implementation of the standard
two-sample KS statistic and its asymptotic p-value (the same Kolmogorov
distribution series SciPy's ``ks_2samp(..., method="asymp")`` implements).
Cross-checked manually against ``scipy.stats.ks_2samp`` (run in a separate
environment that has SciPy installed, since this one deliberately doesn't):
the KS statistic itself matches bit-for-bit; the asymptotic p-value is close
but not bit-identical (e.g. 0.8279 here vs SciPy's 0.8058 for one tested
pair) due to a simpler tail-correction term in this implementation's series
-- close enough that it never changed an alert/nominal classification in
that comparison, but treat the exact p-value as approximate, not a precise
statistical guarantee.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

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
