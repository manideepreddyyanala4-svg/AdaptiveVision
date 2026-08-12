"""Unit tests for M7 metrology inspection."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from adaptivevision.alignment import LocalizedPart
from adaptivevision.common.result import MetrologyResult
from adaptivevision.common.types import MeasurementSpec, Pose, RectifiedFrame, Tolerance
from adaptivevision.inspection.metrology import (
    MetrologyInspector,
    StaticMeasurementSource,
)
from adaptivevision.recipe import Recipe


def _part() -> LocalizedPart:
    frame = RectifiedFrame(
        image=np.ones((4, 4), dtype=np.uint8),
        camera_id="cam0",
        frame_id="frame-1",
        calibration_ver="calib-v1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return LocalizedPart(
        frame=frame,
        pose=Pose(0.0, 0.0, 0.0),
        reference_id="golden",
        reference_ver="ref-v1",
        score=1.0,
    )


def _recipe() -> Recipe:
    return Recipe(
        recipe_id="recipe-1",
        version="v1",
        measurement_specs=(
            MeasurementSpec(
                name="width",
                nominal=10.0,
                tolerance=Tolerance(minus=0.2, plus=0.3),
                unit="mm",
            ),
        ),
    )


def test_metrology_inspector_records_in_tolerance_measurement() -> None:
    inspector = MetrologyInspector(StaticMeasurementSource({"width": 10.1}).measure)
    result = inspector.inspect(_part(), _recipe())
    assert isinstance(result, MetrologyResult)
    assert result.defects == ()
    assert result.measurements[0].name == "width"
    assert result.measurements[0].in_tolerance is True
    assert result.measurements[0].unit == "mm"


def test_metrology_inspector_flags_out_of_tolerance_measurement() -> None:
    inspector = MetrologyInspector(StaticMeasurementSource({"width": 10.5}).measure)
    result = inspector.inspect(_part(), _recipe())
    assert result.measurements[0].in_tolerance is False
    assert len(result.defects) == 1
    assert "outside tolerance" in (result.defects[0].description or "")


def test_metrology_inspector_flags_missing_measurement() -> None:
    inspector = MetrologyInspector(StaticMeasurementSource({}).measure)
    result = inspector.inspect(_part(), _recipe())
    assert result.measurements == ()
    assert len(result.defects) == 1
    assert "Missing measurement" in (result.defects[0].description or "")


def test_static_measurement_source_returns_copy() -> None:
    values = {"width": 10.1}
    source = StaticMeasurementSource(values)
    values["width"] = 99.0
    measured = source.measure(_part(), _recipe())
    assert measured["width"] == 10.1
