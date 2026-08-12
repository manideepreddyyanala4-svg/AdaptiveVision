"""Camera drivers and the null-object strategy (Milestone M3).

The walking skeleton must run end-to-end without physical hardware. Per the
frozen null-object strategy (Architecture Spec v1.0 §19), every seam has a
no-op / synthetic implementation that is injected when no real adapter is
configured.

:class:`NullCameraDriver` is that implementation for the
:class:`~adaptivevision.common.interfaces.CameraDriver` seam: it is always
"healthy", opens and closes without side effects, and produces a synthetic
grayscale frame of the configured size. Real camera backends replace it behind
the same seam in later milestones.
"""

from __future__ import annotations

import numpy as np

from adaptivevision.acquisition.frame import build_frame
from adaptivevision.common.errors import AcquisitionError
from adaptivevision.common.interfaces import CameraDriver
from adaptivevision.common.types import RawFrame
from adaptivevision.config.settings import CameraConfig


class NullCameraDriver(CameraDriver):
    """A synthetic :class:`CameraDriver` used when no real camera is configured.

    Args:
        config: The camera configuration describing the synthetic frame size.
    """

    def __init__(self, config: CameraConfig) -> None:
        """Initialize the driver with a camera configuration."""
        self._config = config
        self._opened = False

    def open(self) -> None:
        """Mark the driver as open (no real device is involved)."""
        self._opened = True

    def close(self) -> None:
        """Mark the driver as closed."""
        self._opened = False

    def capture(self, trigger_id: str | None = None) -> RawFrame:
        """Produce a synthetic grayscale frame.

        Args:
            trigger_id: Identifier of the triggering event, if any.

        Returns:
            A :class:`RawFrame` with a zero-filled image of the configured size.

        Raises:
            AcquisitionError: If the driver is not open.
        """
        if not self._opened:
            msg = "NullCameraDriver.capture called before open()"
            raise AcquisitionError(msg)
        image = np.zeros(
            (self._config.height, self._config.width),
            dtype=np.uint8,
        )
        return build_frame(
            image,
            self._config.camera_id,
            trigger_id=trigger_id,
        )

    def is_healthy(self) -> bool:
        """Return ``True`` while the driver is open."""
        return self._opened
