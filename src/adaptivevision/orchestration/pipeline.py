"""The inspection pipeline (Milestone M3, extended M5/M6/M7).

The pipeline is the heart of the walking skeleton: it drives one inspection
cycle by acquiring a frame from the camera driver and producing an
:class:`~adaptivevision.common.result.InspectionResult`.

M5 adds optional preprocessing and calibration rectification stages. M6 adds an
optional alignment stage that produces a localized part for downstream
inspectors. These stages are injected as callables so orchestration stays
decoupled from concrete implementations.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from adaptivevision.alignment import LocalizedPart
from adaptivevision.common.enums import Verdict
from adaptivevision.common.interfaces import CameraDriver, Inspector
from adaptivevision.common.result import InspectionResult, MetrologyResult
from adaptivevision.common.types import RawFrame, RectifiedFrame
from adaptivevision.recipe import Recipe

Preprocessor = Callable[[RawFrame], RawFrame]
Rectifier = Callable[[RawFrame], RectifiedFrame]
Aligner = Callable[[RectifiedFrame], LocalizedPart]


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

    def __init__(
        self,
        camera: CameraDriver,
        *,
        station_id: str,
        recipe_ver: str,
        preprocessor: Preprocessor | None = None,
        rectifier: Rectifier | None = None,
        aligner: Aligner | None = None,
        recipe: Recipe | None = None,
        metrology_inspector: Inspector[LocalizedPart, Recipe] | None = None,
    ) -> None:
        """Initialize the pipeline."""
        self._camera = camera
        self._station_id = station_id
        self._recipe_ver = recipe_ver
        self._preprocessor = preprocessor
        self._rectifier = rectifier
        self._aligner = aligner
        self._recipe = recipe
        self._metrology_inspector = metrology_inspector

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
        frame = self._preprocess(frame)
        rectified = self._rectify(frame)
        part = self._align(rectified)
        metrology = self._inspect_metrology(part)
        cycle_time_ms = (time.monotonic() - started) * 1000.0
        defects = metrology.defects if metrology is not None else ()
        measurements = metrology.measurements if metrology is not None else ()

        return InspectionResult(
            inspection_id=new_inspection_id(),
            part_id=part_id,
            station_id=self._station_id,
            verdict=Verdict.FAIL if defects else Verdict.PASS,
            recipe_ver=self._recipe_ver,
            model_ver="",
            calib_ver=rectified.calibration_ver,
            cycle_time_ms=cycle_time_ms,
            timestamp_utc=datetime.now(UTC),
            measurements=measurements,
            defects=defects,
            image_refs=(rectified.frame_id,),
        )

    def _acquire(self, trigger_id: str | None) -> RawFrame:
        """Acquire a frame from the camera driver."""
        return self._camera.capture(trigger_id)

    def _preprocess(self, frame: RawFrame) -> RawFrame:
        """Apply the optional preprocessing stage."""
        if self._preprocessor is None:
            return frame
        return self._preprocessor(frame)

    def _rectify(self, frame: RawFrame) -> RectifiedFrame:
        """Apply the optional rectification stage."""
        if self._rectifier is None:
            return RectifiedFrame(
                image=frame.image,
                camera_id=frame.camera_id,
                frame_id=frame.frame_id,
                calibration_ver="",
                timestamp_monotonic=frame.timestamp_monotonic,
                timestamp_utc=frame.timestamp_utc,
                trigger_id=frame.trigger_id,
            )
        return self._rectifier(frame)

    def _align(self, frame: RectifiedFrame) -> LocalizedPart | None:
        """Apply the optional alignment stage."""
        if self._aligner is None:
            return None
        return self._aligner(frame)

    def _inspect_metrology(self, part: LocalizedPart | None) -> MetrologyResult | None:
        """Apply the optional M7 metrology stage."""
        if part is None or self._recipe is None or self._metrology_inspector is None:
            return None
        partial = self._metrology_inspector.inspect(part, self._recipe)
        if not isinstance(partial, MetrologyResult):
            msg = "Metrology inspector must return MetrologyResult"
            raise TypeError(msg)
        return partial
