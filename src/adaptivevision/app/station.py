"""The station controller (Milestone M3).

The station controller is the composition root's orchestrator: it owns the
station state machine and drives the inspection pipeline through the scheduler,
enforcing cycle-time limits with the watchdog. It is the single object the
application entrypoint interacts with.

The controller is deliberately synchronous at M3. Later milestones add
trigger-driven acquisition, multi-threaded scheduling, and richer fault
handling, but the public surface here is the stable contract.
"""

from __future__ import annotations

from collections.abc import Callable

from adaptivevision.common.enums import StationState
from adaptivevision.common.errors import AdaptiveVisionError
from adaptivevision.common.result import InspectionResult
from adaptivevision.orchestration.pipeline import InspectionPipeline
from adaptivevision.orchestration.scheduler import InspectionScheduler
from adaptivevision.orchestration.state import StationStateMachine
from adaptivevision.orchestration.watchdog import CycleWatchdog


class StationController:
    """Coordinates the station lifecycle and inspection cycles.

    Args:
        state_machine: The station state machine.
        pipeline: The inspection pipeline.
        scheduler: The inspection scheduler.
        watchdog: The cycle watchdog.
        on_result: Optional callback invoked with each result as it is produced
            (used to persist results off the critical path).
    """

    def __init__(
        self,
        state_machine: StationStateMachine,
        pipeline: InspectionPipeline,
        scheduler: InspectionScheduler,
        watchdog: CycleWatchdog,
        on_result: Callable[[InspectionResult], None] | None = None,
    ) -> None:
        """Initialize the controller with its collaborators."""
        self._state = state_machine
        self._pipeline = pipeline
        self._scheduler = scheduler
        self._watchdog = watchdog
        self._on_result = on_result

    @property
    def state(self) -> StationState:
        """Return the current station state."""
        return self._state.state

    def boot(self) -> None:
        """Run the boot sequence: ``INIT -> SELF_TEST -> IDLE``.

        Raises:
            FaultError: If a transition is invalid.
        """
        self._state.transition(StationState.SELF_TEST)
        self._state.transition(StationState.IDLE)

    def ready(self) -> None:
        """Transition the station to ``READY``.

        Raises:
            FaultError: If the transition is invalid.
        """
        self._state.transition(StationState.READY)

    def run(self, part_ids: list[str]) -> tuple[InspectionResult, ...]:
        """Inspect a batch of parts.

        Transitions to ``RUNNING``, runs one cycle per part, then returns to
        ``READY``.

        Args:
            part_ids: Identifiers of the parts to inspect.

        Returns:
            A tuple of inspection results, one per part.

        Raises:
            FaultError: If the station is not in a state that can run.
            AdaptiveVisionError: If a cycle fails.
        """
        self._state.transition(StationState.RUNNING)
        try:
            results = self._scheduler.run_cycles(part_ids, on_result=self._on_result)
        except AdaptiveVisionError:
            self._state.to_fault()
            raise
        finally:
            if self._state.state is StationState.RUNNING:
                self._state.transition(StationState.READY)
        return results

    def shutdown(self) -> None:
        """Transition the station to ``SHUTDOWN``.

        Raises:
            FaultError: If the transition is invalid.
        """
        self._state.transition(StationState.SHUTDOWN)
