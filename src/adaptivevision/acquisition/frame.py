"""Frame grabbing and construction (Milestone M3).

This module introduces the runtime NumPy dependency (frozen decision 9: the
frame image type is resolved at runtime from Milestone M3 onward) and provides
helpers to build :class:`~adaptivevision.common.types.RawFrame` objects with
correct acquisition metadata.

The acquisition layer is deliberately thin at M3: it produces frames and hands
them to the orchestration pipeline. Real camera backends (GigE Vision, USB3,
GenICam) are injected behind the :class:`~adaptivevision.common.interfaces.CameraDriver`
seam in later milestones.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from adaptivevision.common.types import RawFrame

if TYPE_CHECKING:
    from adaptivevision.common.types import Image


def new_frame_id() -> str:
    """Return a unique frame identifier."""
    return f"frame-{uuid.uuid4().hex[:12]}"


def build_frame(
    image: Image,
    camera_id: str,
    *,
    trigger_id: str | None = None,
    frame_id: str | None = None,
) -> RawFrame:
    """Build a :class:`RawFrame` with populated acquisition metadata.

    Args:
        image: The raw image buffer.
        camera_id: Identifier of the source camera.
        trigger_id: Identifier of the triggering event, if any.
        frame_id: Explicit frame identifier; a new one is generated if omitted.

    Returns:
        A :class:`RawFrame` carrying the image and its metadata.
    """
    return RawFrame(
        image=image,
        camera_id=camera_id,
        frame_id=frame_id or new_frame_id(),
        timestamp_monotonic=time.monotonic(),
        timestamp_utc=datetime.now(UTC),
        trigger_id=trigger_id,
    )
