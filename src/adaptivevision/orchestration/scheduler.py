"""The inspection scheduler (Milestone M3).

The scheduler drives the station through inspection cycles while it is in the
``RUNNING`` state. At M3 it is a simple, synchronous driver: it runs a bounded
number of cycles against the pipeline and reports each result. Later milestones
replace this with a trigger-driven, multi-threaded scheduler.

The scheduler is deliberately decoupled from the state machine: it only runs
cycles and returns results. The station controller (composition root) owns the
state transitions around it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from adaptivevision.common.result import InspectionResult
from adaptivevision.orchestration.pipeline import InspectionPipeline


class InspectionScheduler:
    """Runs inspection cycles against a pipeline.

    Args:
        pipeline: The pipeline to drive.
    """

    def __init__(self, pipeline: InspectionPipeline) -> None:
        """Initialize the scheduler."""
        self._pipeline = pipeline

    def run_cycles(
        self,
        part_ids: Iterable[str],
        *,
        on_result: Callable[[InspectionResult], None] | None = None,
    ) -> tuple[InspectionResult, ...]:
        """Run one inspection cycle per part id.

        Args:
            part_ids: Identifiers of the parts to inspect.
            on_result: Optional callback invoked with each result as it is
                produced.

        Returns:
            A tuple of inspection results, one per part id.
        """
        results: list[InspectionResult] = []
        for part_id in part_ids:
            result = self._pipeline.run(part_id)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return tuple(results)
