"""Timing primitives for latency measurement and deadline tracking.

Per frozen decision 7, durations and deadlines use :func:`time.monotonic`,
which is immune to wall-clock adjustments (NTP, DST). These helpers only
*measure* time; enforcement of the latency budget is the watchdog's job
(Milestone M3).

Every helper accepts an injectable ``clock`` callable so tests are
deterministic and never sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

Clock = Callable[[], float]


class Stopwatch:
    """Measures elapsed monotonic time from a start point."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        """Start the stopwatch.

        Args:
            clock: Monotonic time source in seconds. Defaults to
                :func:`time.monotonic`.
        """
        self._clock = clock
        self._start = clock()

    def reset(self) -> None:
        """Restart the stopwatch from the current time."""
        self._start = self._clock()

    def elapsed_s(self) -> float:
        """Return elapsed time in seconds."""
        return self._clock() - self._start

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed_s() * 1000.0


class Deadline:
    """Tracks a fixed time budget measured against a monotonic clock."""

    def __init__(self, budget_s: float, clock: Clock = time.monotonic) -> None:
        """Create a deadline ``budget_s`` seconds from now.

        Args:
            budget_s: Time budget in seconds.
            clock: Monotonic time source in seconds. Defaults to
                :func:`time.monotonic`.
        """
        self._clock = clock
        self._deadline = clock() + budget_s

    @classmethod
    def from_ms(cls, budget_ms: float, clock: Clock = time.monotonic) -> Deadline:
        """Create a deadline from a budget expressed in milliseconds."""
        return cls(budget_ms / 1000.0, clock=clock)

    def remaining_s(self) -> float:
        """Return remaining time in seconds (negative once expired)."""
        return self._deadline - self._clock()

    def expired(self) -> bool:
        """Return ``True`` once the budget has elapsed."""
        return self._clock() >= self._deadline


@contextmanager
def measure(clock: Clock = time.monotonic) -> Iterator[Stopwatch]:
    """Context manager yielding a :class:`Stopwatch` for the enclosed block.

    Args:
        clock: Monotonic time source in seconds. Defaults to
            :func:`time.monotonic`.

    Yields:
        A running :class:`Stopwatch`; read ``elapsed_ms()`` after the block.
    """
    stopwatch = Stopwatch(clock)
    yield stopwatch
