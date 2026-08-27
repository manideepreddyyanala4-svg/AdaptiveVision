"""In-memory result buffering (Milestone M17).

The :class:`ResultBuffer` holds inspection results that could not be persisted
immediately so they can be retried later without losing data.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from adaptivevision.common.result import InspectionResult


class ResultBuffer:
    """A bounded FIFO buffer of inspection results awaiting persistence."""

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize an empty buffer with a maximum ``capacity``."""
        if capacity <= 0:
            msg = "ResultBuffer capacity must be positive"
            raise ValueError(msg)
        self._capacity = capacity
        self._items: deque[InspectionResult] = deque()

    def push(self, result: InspectionResult) -> None:
        """Append a result, dropping the oldest if the buffer is full."""
        self._items.append(result)
        if len(self._items) > self._capacity:
            self._items.popleft()

    def drain(self) -> tuple[InspectionResult, ...]:
        """Remove and return all buffered results."""
        items = tuple(self._items)
        self._items.clear()
        return items

    def __len__(self) -> int:
        """Return the number of buffered results."""
        return len(self._items)

    def is_full(self) -> bool:
        """Return ``True`` if the buffer is at capacity."""
        return len(self._items) >= self._capacity

    def extend(self, results: Iterable[InspectionResult]) -> None:
        """Append multiple results."""
        for result in results:
            self.push(result)
