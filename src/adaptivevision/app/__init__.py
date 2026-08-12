"""Application composition root (Milestone M3).

This package wires the walking skeleton together: :func:`build_station`
assembles the camera driver, pipeline, scheduler, watchdog, and state machine
into a :class:`StationController` from validated configuration. It is the only
place concrete implementations are bound to the abstraction seams.
"""

from __future__ import annotations

from adaptivevision.app.app import build_camera, build_station
from adaptivevision.app.station import StationController

__all__ = [
    "StationController",
    "build_camera",
    "build_station",
]
