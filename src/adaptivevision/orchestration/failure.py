"""Failure handling with retry and buffering (Milestone M17).

The :class:`FailureHandler` wraps a persistence callable and retries it a
bounded number of times. If the callable still fails, the result is buffered
for a later retry so no inspection result is lost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from adaptivevision.common.result import InspectionResult
from adaptivevision.orchestration.buffer import ResultBuffer


@dataclass(frozen=True, slots=True)
class FailureOutcome:
    """Outcome of a persistence attempt.

    Attributes:
        persisted: Whether the result was persisted successfully.
        attempts: Number of attempts made.
        buffered: Whether the result was buffered for a later retry.
    """

    persisted: bool
    attempts: int
    buffered: bool


class FailureHandler:
    """Retries persistence and buffers results that still fail."""

    def __init__(
        self,
        persist: Callable[[InspectionResult], None],
        *,
        max_attempts: int = 3,
        buffer: ResultBuffer | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            persist: Callable that persists a single result.
            max_attempts: Maximum number of attempts before buffering.
            buffer: Buffer for results that could not be persisted.
        """
        if max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)
        self._persist = persist
        self._max_attempts = max_attempts
        self._buffer = buffer if buffer is not None else ResultBuffer()

    def handle(self, result: InspectionResult) -> FailureOutcome:
        """Persist ``result`` with retries, buffering on persistent failure.

        Args:
            result: The inspection result to persist.

        Returns:
            The outcome of the persistence attempt.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._persist(result)
                return FailureOutcome(persisted=True, attempts=attempt, buffered=False)
            except Exception:
                if attempt == self._max_attempts:
                    self._buffer.push(result)
                    return FailureOutcome(
                        persisted=False, attempts=attempt, buffered=True
                    )
        # Unreachable; kept for type-checker completeness.
        return FailureOutcome(
            persisted=False, attempts=self._max_attempts, buffered=True
        )

    def flush(self) -> tuple[InspectionResult, ...]:
        """Attempt to persist all buffered results.

        Returns:
            The results that still could not be persisted.
        """
        remaining: list[InspectionResult] = []
        for result in self._buffer.drain():
            outcome = self.handle(result)
            if not outcome.persisted:
                remaining.append(result)
        return tuple(remaining)
