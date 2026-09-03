"""Unit tests for M6 alignment."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from adaptivevision.camera import (
    GoldenReference,
    ReferenceAligner,
    load_golden_reference,
)
from adaptivevision.common import FaultError
from adaptivevision.common import Pose, RectifiedFrame


def _reference_data() -> dict[str, object]:
    return {
        "reference_id": "golden-1",
        "version": "ref-v1",
        "camera_id": "cam0",
        "image_width": 4,
        "image_height": 3,
        "nominal_pose": {"x": 1.0, "y": 2.0, "theta_deg": 3.0},
        "min_score": 0.9,
    }


def _frame(camera_id: str = "cam0", shape: tuple[int, int] = (3, 4)) -> RectifiedFrame:
    return RectifiedFrame(
        image=np.ones(shape, dtype=np.uint8),
        camera_id=camera_id,
        frame_id="frame-1",
        calibration_ver="calib-v1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_golden_reference_roundtrip() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    assert GoldenReference.from_dict(reference.to_dict()) == reference


def test_golden_reference_defaults_nominal_pose() -> None:
    data = _reference_data()
    data.pop("nominal_pose")
    reference = GoldenReference.from_dict(data)
    assert reference.nominal_pose == Pose(0.0, 0.0, 0.0)


def test_golden_reference_rejects_invalid_score() -> None:
    data = _reference_data()
    data["min_score"] = 1.5
    with pytest.raises(FaultError, match=r"\[0, 1\]"):
        GoldenReference.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_id", "", "reference_id"),
        ("version", "", "version"),
        ("camera_id", "", "camera_id"),
        ("image_width", 0, "dimensions"),
    ],
)
def test_golden_reference_rejects_invalid_required_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    data = _reference_data()
    data[field] = value
    with pytest.raises(FaultError, match=message):
        GoldenReference.from_dict(data)


def test_load_golden_reference_from_json(tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference_data()), encoding="utf-8")
    assert load_golden_reference(path).version == "ref-v1"


def test_load_golden_reference_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FaultError, match="Failed to load golden reference"):
        load_golden_reference(tmp_path / "missing.json")


def test_load_golden_reference_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(FaultError, match="must contain a JSON object"):
        load_golden_reference(path)


def test_load_golden_reference_rejects_invalid_artifact(tmp_path) -> None:
    path = tmp_path / "bad-reference.json"
    data = _reference_data()
    data.pop("reference_id")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FaultError, match="Invalid golden reference"):
        load_golden_reference(path)


def test_reference_aligner_returns_localized_part() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    part = ReferenceAligner(reference).align(_frame())
    assert part.reference_id == "golden-1"
    assert part.reference_ver == "ref-v1"
    assert part.pose == Pose(1.0, 2.0, 3.0)
    assert part.score == 1.0
    assert part.lineage()["calibration_ver"] == "calib-v1"


def test_reference_aligner_rejects_camera_mismatch() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    with pytest.raises(FaultError, match="does not match"):
        ReferenceAligner(reference).align(_frame(camera_id="other"))


def test_reference_aligner_rejects_dimension_mismatch() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    with pytest.raises(FaultError, match="do not match"):
        ReferenceAligner(reference).align(_frame(shape=(2, 4)))


def test_reference_aligner_rejects_min_score_above_estimate() -> None:
    data = _reference_data()
    data["min_score"] = 1.0
    reference = GoldenReference.from_dict(data)
    assert ReferenceAligner(reference).align(_frame()).score == 1.0


def test_localized_part_rejects_invalid_score() -> None:
    frame = _frame()
    with pytest.raises(FaultError, match=r"\[0, 1\]"):
        from adaptivevision.camera import LocalizedPart

        LocalizedPart(
            frame=frame,
            pose=Pose(0.0, 0.0, 0.0),
            reference_id="golden",
            reference_ver="ref-v1",
            score=1.1,
        )
