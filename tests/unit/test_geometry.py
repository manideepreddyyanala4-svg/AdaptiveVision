"""Unit tests for :mod:`adaptivevision.common.geometry`."""

from __future__ import annotations

import math

import pytest

from adaptivevision.common import geometry


def test_deg_rad_roundtrip() -> None:
    assert geometry.deg_to_rad(180.0) == pytest.approx(math.pi)
    assert geometry.rad_to_deg(math.pi) == pytest.approx(180.0)


def test_normalize_angle_deg() -> None:
    assert geometry.normalize_angle_deg(190.0) == pytest.approx(-170.0)
    assert geometry.normalize_angle_deg(-190.0) == pytest.approx(170.0)
    assert geometry.normalize_angle_deg(0.0) == pytest.approx(0.0)
    assert geometry.normalize_angle_deg(180.0) == pytest.approx(-180.0)


def test_distance() -> None:
    assert geometry.distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_angle_between_deg() -> None:
    assert geometry.angle_between_deg((0.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)


def test_rotate_point_about_origin() -> None:
    x, y = geometry.rotate_point((1.0, 0.0), 90.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.0)


def test_rotate_point_about_custom_origin() -> None:
    x, y = geometry.rotate_point((2.0, 1.0), 90.0, origin=(1.0, 1.0))
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)


def test_translate_point() -> None:
    assert geometry.translate_point((1.0, 2.0), 3.0, -1.0) == (4.0, 1.0)


def test_transform_point_rotate_then_translate() -> None:
    x, y = geometry.transform_point((10.0, 5.0, 90.0), (1.0, 0.0))
    assert x == pytest.approx(10.0, abs=1e-9)
    assert y == pytest.approx(6.0)


def test_compose_matches_sequential_transform() -> None:
    first = (1.0, 2.0, 30.0)
    second = (-3.0, 4.0, 45.0)
    point = (5.0, -2.0)
    composed = geometry.compose_pose(first, second)
    via_compose = geometry.transform_point(composed, point)
    via_sequential = geometry.transform_point(second, geometry.transform_point(first, point))
    assert via_compose[0] == pytest.approx(via_sequential[0])
    assert via_compose[1] == pytest.approx(via_sequential[1])


def test_invert_pose_is_inverse() -> None:
    pose = (3.0, -1.5, 37.0)
    identity = geometry.compose_pose(pose, geometry.invert_pose(pose))
    assert identity[0] == pytest.approx(0.0, abs=1e-9)
    assert identity[1] == pytest.approx(0.0, abs=1e-9)
    assert identity[2] == pytest.approx(0.0, abs=1e-9)


def test_apply_homography_identity() -> None:
    identity: geometry.Homography = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert geometry.apply_homography(identity, (2.0, 3.0)) == pytest.approx((2.0, 3.0))


def test_apply_homography_scaling() -> None:
    scale: geometry.Homography = (
        (2.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    x, y = geometry.apply_homography(scale, (4.0, 5.0))
    assert x == pytest.approx(8.0)
    assert y == pytest.approx(15.0)


def test_apply_homography_raises_on_degenerate() -> None:
    degenerate: geometry.Homography = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="Degenerate homography"):
        geometry.apply_homography(degenerate, (1.0, 1.0))
