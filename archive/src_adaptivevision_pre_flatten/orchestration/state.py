"""Station state machine (Milestone M3).

The station lifecycle is a finite state machine over
:class:`~adaptivevision.common.enums.StationState` (Architecture Spec v1.0
§18). This module owns the transition table and enforces it: an invalid
transition raises :class:`~adaptivevision.common.errors.FaultError`, which is
non-recoverable and drives the station to a fault / safe state.

The state machine is deliberately small at M3 - it models the boot path
(``INIT -> SELF_TEST -> IDLE -> READY -> RUNNING``) and the fault / shutdown
paths. Later milestones extend the table with calibration, recipe loading,
pause, and maintenance transitions.
"""

from __future__ import annotations

from adaptivevision.common.enums import StationState
from adaptivevision.common.errors import FaultError

#: Allowed transitions, keyed by source state.
_TRANSITIONS: dict[StationState, frozenset[StationState]] = {
    StationState.INIT: frozenset(
        {StationState.SELF_TEST, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.SELF_TEST: frozenset(
        {StationState.IDLE, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.IDLE: frozenset(
        {
            StationState.READY,
            StationState.CALIBRATION,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.CALIBRATION: frozenset(
        {StationState.IDLE, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.RECIPE_LOADING: frozenset(
        {StationState.READY, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.READY: frozenset(
        {
            StationState.RUNNING,
            StationState.IDLE,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.RUNNING: frozenset(
        {
            StationState.READY,
            StationState.PAUSED,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.PAUSED: frozenset(
        {
            StationState.RUNNING,
            StationState.READY,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.FAULT: frozenset(
        {StationState.MAINTENANCE, StationState.ESTOP, StationState.SHUTDOWN}
    ),
    StationState.ESTOP: frozenset({StationState.SHUTDOWN}),
    StationState.MAINTENANCE: frozenset(
        {StationState.IDLE, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.SHUTDOWN: frozenset(),
}


class StationStateMachine:
    """A guarded finite state machine over :class:`StationState`.

    Args:
        initial: The initial state. Defaults to :attr:`StationState.INIT`.
    """

    def __init__(self, initial: StationState = StationState.INIT) -> None:
        """Initialize the state machine."""
        self._state = initial

    @property
    def state(self) -> StationState:
        """Return the current state."""
        return self._state

    def can_transition(self, target: StationState) -> bool:
        """Return ``True`` if a transition to ``target`` is allowed."""
        return target in _TRANSITIONS[self._state]

    def transition(self, target: StationState) -> StationState:
        """Transition to ``target``, enforcing the transition table.

        Args:
            target: The state to transition to.

        Returns:
            The new state.

        Raises:
            FaultError: If the transition is not allowed from the current state.
        """
        if not self.can_transition(target):
            msg = f"Invalid station transition: {self._state.value} -> {target.value}"
            raise FaultError(msg)
        self._state = target
        return self._state

    def to_fault(self) -> StationState:
        """Transition to :attr:`StationState.FAULT` if allowed.

        Returns:
            The new state (``FAULT`` if the transition succeeded, otherwise the
            current state unchanged).
        """
        if self.can_transition(StationState.FAULT):
            return self.transition(StationState.FAULT)
        return self._state
