"""Pure-Python 2D geometry primitives.

Backend-agnostic helpers for the coordinate math used by alignment (Milestone
M6) and metrology (Milestone M7). Per frozen decision 3, this module has no
third-party dependencies - no NumPy, no OpenCV - and operates on plain tuples
and floats so it is deterministic and trivially testable.

A 2D rigid pose is represented as ``(x, y, theta_deg)`` and applies as
``point -> R(theta_deg) @ point + (x, y)`` (rotate about the origin, then
translate).
"""

from __future__ import annotations

import math

#: A 2D point as ``(x, y)``.
Point = tuple[float, float]

#: A 2D rigid pose as ``(x, y, theta_deg)``.
PoseTuple = tuple[float, float, float]

#: A 3x3 homography as a tuple of three row tuples.
Homography = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def normalize_angle_deg(degrees: float) -> float:
    """Normalize an angle to the half-open range ``[-180, 180)`` degrees."""
    return (degrees + 180.0) % 360.0 - 180.0


def distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_between_deg(a: Point, b: Point) -> float:
    """Return the angle in degrees of the vector from ``a`` to ``b``."""
    return rad_to_deg(math.atan2(b[1] - a[1], b[0] - a[0]))


def rotate_point(point: Point, degrees: float, origin: Point = (0.0, 0.0)) -> Point:
    """Rotate ``point`` about ``origin`` by ``degrees`` (counter-clockwise)."""
    rad = deg_to_rad(degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    x = cos_a * dx - sin_a * dy + origin[0]
    y = sin_a * dx + cos_a * dy + origin[1]
    return (x, y)


def translate_point(point: Point, dx: float, dy: float) -> Point:
    """Translate ``point`` by ``(dx, dy)``."""
    return (point[0] + dx, point[1] + dy)


def transform_point(pose: PoseTuple, point: Point) -> Point:
    """Apply a rigid ``pose`` to ``point`` (rotate about origin, then translate)."""
    px, py, theta = pose
    rotated = rotate_point(point, theta)
    return (rotated[0] + px, rotated[1] + py)


def compose_pose(first: PoseTuple, second: PoseTuple) -> PoseTuple:
    """Compose two poses: the result applies ``first`` and then ``second``.

    For any point ``p``, ``transform_point(compose_pose(first, second), p)``
    equals ``transform_point(second, transform_point(first, p))``.

    Args:
        first: The pose applied first.
        second: The pose applied second.

    Returns:
        The composed pose.
    """
    rotated = rotate_point((first[0], first[1]), second[2])
    x = rotated[0] + second[0]
    y = rotated[1] + second[1]
    theta = normalize_angle_deg(first[2] + second[2])
    return (x, y, theta)


def invert_pose(pose: PoseTuple) -> PoseTuple:
    """Return the inverse of a rigid ``pose``.

    ``compose_pose(pose, invert_pose(pose))`` is the identity pose
    ``(0, 0, 0)`` up to floating-point error.
    """
    x, y, theta = pose
    rx, ry = rotate_point((x, y), -theta)
    return (-rx, -ry, normalize_angle_deg(-theta))


def apply_homography(homography: Homography, point: Point) -> Point:
    """Apply a 3x3 ``homography`` to ``point`` and return the projected point.

    Args:
        homography: A 3x3 projective transform.
        point: The point to project.

    Returns:
        The projected point.

    Raises:
        ValueError: If the projective denominator is zero (degenerate mapping).
    """
    x, y = point
    row0, row1, row2 = homography
    w = row2[0] * x + row2[1] * y + row2[2]
    if w == 0.0:
        msg = "Degenerate homography: projective denominator is zero"
        raise ValueError(msg)
    px = (row0[0] * x + row0[1] * y + row0[2]) / w
    py = (row1[0] * x + row1[1] * y + row1[2]) / w
    return (px, py)
