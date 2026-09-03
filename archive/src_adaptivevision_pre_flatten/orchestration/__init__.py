"""State machine, pipeline, scheduler, watchdog (Milestone M3).

This package implements the orchestration layer of the walking skeleton: the
station :class:`StationStateMachine`, the :class:`InspectionPipeline` that runs
one inspection cycle, the :class:`InspectionScheduler` that drives cycles, and
the :class:`CycleWatchdog` that enforces cycle-time limits.
"""

from __future__ import annotations

from adaptivevision.orchestration.buffer import ResultBuffer
from adaptivevision.orchestration.failure import FailureHandler, FailureOutcome
from adaptivevision.orchestration.pipeline import InspectionPipeline, new_inspection_id
from adaptivevision.orchestration.scheduler import InspectionScheduler
from adaptivevision.orchestration.state import StationStateMachine
from adaptivevision.orchestration.watchdog import CycleWatchdog

__all__ = [
    "CycleWatchdog",
    "FailureHandler",
    "FailureOutcome",
    "InspectionPipeline",
    "InspectionScheduler",
    "ResultBuffer",
    "StationStateMachine",
    "new_inspection_id",
]
