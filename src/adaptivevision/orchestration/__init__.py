"""State machine, pipeline, scheduler, watchdog (Milestone M3).

This package implements the orchestration layer of the walking skeleton: the
station :class:`StationStateMachine`, the :class:`InspectionPipeline` that runs
one inspection cycle, the :class:`InspectionScheduler` that drives cycles, and
the :class:`CycleWatchdog` that enforces cycle-time limits.
"""

from __future__ import annotations

from adaptivevision.orchestration.pipeline import InspectionPipeline, new_inspection_id
from adaptivevision.orchestration.scheduler import InspectionScheduler
from adaptivevision.orchestration.state import StationStateMachine
from adaptivevision.orchestration.watchdog import CycleWatchdog

__all__ = [
    "CycleWatchdog",
    "InspectionPipeline",
    "InspectionScheduler",
    "StationStateMachine",
    "new_inspection_id",
]
