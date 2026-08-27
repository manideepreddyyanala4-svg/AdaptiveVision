"""Unit tests for :mod:`adaptivevision.common.types`."""

from __future__ import annotations

import dataclasses

import pytest

from adaptivevision.common import types


def test_roi_center_and_validation() -> None:
    roi = types.ROI(label="pad", x=10.0, y=20.0, width=4.0, height=6.0)
    assert roi.center == (12.0, 23.0)


def test_roi_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        types.ROI(label="bad", x=0.0, y=0.0, width=-1.0, height=1.0)


def test_roi_roundtrip() -> None:
    roi = types.ROI(label="a", x=1.0, y=2.0, width=3.0, height=4.0, angle_deg=5.0)
    assert types.ROI.from_dict(roi.to_dict()) == roi


def test_roi_is_frozen() -> None:
    roi = types.ROI(label="a", x=0.0, y=0.0, width=1.0, height=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        roi.x = 9.0  # type: ignore[misc]


def test_pose_compose_inverse_transform() -> None:
    pose = types.Pose(3.0, -1.0, 90.0)
    identity = pose.compose(pose.inverse())
    assert identity.x == pytest.approx(0.0, abs=1e-9)
    assert identity.y == pytest.approx(0.0, abs=1e-9)
    assert identity.theta_deg == pytest.approx(0.0, abs=1e-9)
    tx, ty = pose.transform_point((1.0, 0.0))
    assert (tx, ty) == pytest.approx((3.0, 0.0), abs=1e-9)


def test_pose_roundtrip_and_as_tuple() -> None:
    pose = types.Pose(1.0, 2.0, 30.0)
    assert pose.as_tuple() == (1.0, 2.0, 30.0)
    assert types.Pose.from_dict(pose.to_dict()) == pose


def test_tolerance_validation() -> None:
    with pytest.raises(ValueError, match="minus"):
        types.Tolerance(minus=-1.0, plus=1.0)
    with pytest.raises(ValueError, match="plus"):
        types.Tolerance(minus=1.0, plus=-1.0)


def test_tolerance_roundtrip() -> None:
    tol = types.Tolerance(minus=0.1, plus=None)
    assert types.Tolerance.from_dict(tol.to_dict()) == tol


def test_measurement_spec_bounds_and_contains() -> None:
    spec = types.MeasurementSpec(
        name="width",
        nominal=10.0,
        tolerance=types.Tolerance(minus=0.2, plus=0.3),
        unit="mm",
    )
    assert spec.lower == pytest.approx(9.8)
    assert spec.upper == pytest.approx(10.3)
    assert spec.contains(9.8) is True
    assert spec.contains(10.3) is True
    assert spec.contains(9.79) is False
    assert spec.contains(10.31) is False


def test_measurement_spec_unbounded_sides() -> None:
    spec = types.MeasurementSpec(
        name="gap",
        nominal=5.0,
        tolerance=types.Tolerance(minus=None, plus=1.0),
        unit="mm",
    )
    assert spec.lower is None
    assert spec.contains(-100.0) is True
    assert spec.contains(6.0) is True
    assert spec.contains(6.1) is False


def test_measurement_spec_unbounded_above() -> None:
    spec = types.MeasurementSpec(
        name="clearance",
        nominal=5.0,
        tolerance=types.Tolerance(minus=1.0, plus=None),
        unit="mm",
    )
    assert spec.upper is None
    assert spec.contains(1_000.0) is True
    assert spec.contains(4.0) is True
    assert spec.contains(3.9) is False


def test_measurement_spec_roundtrip() -> None:
    spec = types.MeasurementSpec(
        name="d",
        nominal=1.0,
        tolerance=types.Tolerance(minus=0.1, plus=0.1),
        unit="mm",
    )
    assert types.MeasurementSpec.from_dict(spec.to_dict()) == spec


def test_measurement_roundtrip_with_and_without_spec() -> None:
    spec = types.MeasurementSpec(
        name="w",
        nominal=2.0,
        tolerance=types.Tolerance(minus=0.1, plus=0.1),
        unit="mm",
    )
    with_spec = types.Measurement(
        name="w", value=2.05, unit="mm", spec=spec, in_tolerance=True
    )
    without_spec = types.Measurement(name="raw", value=3.0, unit="px")
    assert types.Measurement.from_dict(with_spec.to_dict()) == with_spec
    assert types.Measurement.from_dict(without_spec.to_dict()) == without_spec


def test_frame_image_excluded_from_equality() -> None:
    from datetime import UTC, datetime

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    a = types.RawFrame(
        image=object(),
        camera_id="cam0",
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=ts,
    )
    b = types.RawFrame(
        image=object(),  # different object, but excluded from compare
        camera_id="cam0",
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=ts,
    )
    assert a == b


def test_rectified_frame_construction() -> None:
    from datetime import UTC, datetime

    frame = types.RectifiedFrame(
        image=object(),
        camera_id="cam0",
        frame_id="frame-1",
        calibration_ver="calib-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert frame.calibration_ver == "calib-1"
    assert frame.trigger_id is None
