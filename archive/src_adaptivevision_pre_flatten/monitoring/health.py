"""Component health checks (Milestone M14).

A :class:`HealthCheck` aggregates the health status of named components so the
station can report an overall health summary to the dashboard and monitoring
tooling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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
        return tuple(
            ComponentStatus(name=name, healthy=probe())
            for name, probe in self._probes.items()
        )

    def is_healthy(self) -> bool:
        """Return ``True`` if every registered component is healthy."""
        return all(status.healthy for status in self.check())
