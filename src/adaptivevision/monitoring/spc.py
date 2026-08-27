"""Statistical process control (SPC) helpers (Milestone M14).

Provides control-chart calculations for monitoring a measured feature over
time: mean, standard deviation, control limits, and out-of-control detection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


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
