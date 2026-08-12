"""The inspection pipeline (Milestone M3).

The pipeline is the heart of the walking skeleton: it drives one inspection
cycle by acquiring a frame from the camera driver and producing an
:class:`~adaptivevision.common.result.InspectionResult`.

At M3 the pipeline is deliberately minimal - it acquires a frame and emits a
placeholder result with no inspectors (those arrive at M7+). The structure is
the stable contract the scheduler drives and later milestones extend with
rectification, alignment, metrology, and anomaly stages.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from adaptivevision.common.enums import Verdict
from adaptivevision.common.interfaces import CameraDriver
from adaptivevision.common.result import InspectionResult
from adaptivevision.common.types import RawFrame


def new_inspection_id() -> str:
    """Return a unique inspection identifier."""
    return f"inspection-{uuid.uuid4().hex[:12]}"


class InspectionPipeline:
    """Runs a single inspection cycle against a camera driver.

    Args:
        camera: The camera driver to acquire frames from.
        station_id: Identifier of the owning station.
        recipe_ver: Version of the active recipe, for traceability.
    """

    def __init__(self, camera: CameraDriver, *, station_id: str, recipe_ver: str) -> None:
        """Initialize the pipeline."""
        self._camera = camera
        self._station_id = station_id
        self._recipe_ver = recipe_ver

    def run(self, part_id: str, *, trigger_id: str | None = None) -> InspectionResult:
        """Execute one inspection cycle.

        Args:
            part_id: Identifier of the part being inspected.
            trigger_id: Identifier of the triggering event, if any.

        Returns:
            The inspection result for the part.

        Raises:
            AcquisitionError: If the frame cannot be acquired.
        """
        started = time.monotonic()
        frame = self._acquire(trigger_id)
        cycle_time_ms = (time.monotonic() - started) * 1000.0

        return InspectionResult(
            inspection_id=new_inspection_id(),
            part_id=part_id,
            station_id=self._station_id,
            verdict=Verdict.PASS,
            recipe_ver=self._recipe_ver,
            model_ver="",
            calib_ver="",
            cycle_time_ms=cycle_time_ms,
            timestamp_utc=datetime.now(UTC),
            image_refs=(frame.frame_id,),
        )

    def _acquire(self, trigger_id: str | None) -> RawFrame:
        """Acquire a frame from the camera driver."""
        return self._camera.capture(trigger_id)
