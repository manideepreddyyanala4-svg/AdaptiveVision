"""The cycle watchdog (Milestone M3).

The watchdog enforces the maximum allowed inspection cycle time. A cycle that
exceeds the configured timeout is a recoverable fault: it degrades the part to
``REVIEW`` and signals the station to pause, rather than halting the process.

At M3 the watchdog is a simple, synchronous checker invoked by the scheduler
after each cycle. Later milestones move it onto a dedicated monitoring thread.
"""

from __future__ import annotations

from adaptivevision.common.result import InspectionResult


class CycleWatchdog:
    """Monitors inspection cycle times against a configured timeout.

    Args:
        timeout_ms: Maximum allowed cycle time in milliseconds.
    """

    def __init__(self, timeout_ms: float) -> None:
        """Initialize the watchdog with a timeout."""
        self._timeout_ms = timeout_ms
        self._violations = 0

    @property
    def timeout_ms(self) -> float:
        """Return the configured timeout in milliseconds."""
        return self._timeout_ms

    @property
    def violations(self) -> int:
        """Return the number of timeout violations observed."""
        return self._violations

    def check(self, result: InspectionResult) -> bool:
        """Check a result against the timeout.

        Args:
            result: The inspection result to check.

        Returns:
            ``True`` if the cycle exceeded the timeout (a violation), ``False``
            otherwise.
        """
        if result.cycle_time_ms > self._timeout_ms:
            self._violations += 1
            return True
        return False
