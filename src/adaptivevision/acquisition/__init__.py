"""Image acquisition: cameras, triggers, frame grabbing (Milestone M3).

This package implements the acquisition layer of the walking skeleton: the
:class:`NullCameraDriver` null-object for the
:class:`~adaptivevision.common.interfaces.CameraDriver` seam and frame-grabbing
helpers that build :class:`~adaptivevision.common.types.RawFrame` objects. It
introduces the runtime NumPy dependency (frozen decision 9).

Real camera backends (GigE Vision, USB3, GenICam) are injected behind the same
seam in later milestones.
"""

from __future__ import annotations

from adaptivevision.acquisition.camera import NullCameraDriver
from adaptivevision.acquisition.frame import build_frame, new_frame_id
from adaptivevision.acquisition.threaded import ThreadedFrameBuffer

__all__ = [
    "NullCameraDriver",
    "ThreadedFrameBuffer",
    "build_frame",
    "new_frame_id",
]
