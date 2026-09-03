"""Golden-reference alignment (Milestone M6)."""

from __future__ import annotations

from adaptivevision.alignment.model import GoldenReference, LocalizedPart
from adaptivevision.common.errors import FaultError
from adaptivevision.common.types import RectifiedFrame


class ReferenceAligner:
    """Localize a rectified frame against a versioned golden reference.

    M6 establishes the alignment contract and lineage. The default estimator is
    deterministic and conservative: it validates that the frame matches the
    reference camera and dimensions, then emits the reference's nominal pose
    with a perfect score. Rich feature/template matching can replace this
    estimator behind the same callable boundary without changing downstream
    contracts.
    """

    def __init__(self, reference: GoldenReference) -> None:
        """Initialize the aligner."""
        self._reference = reference

    @property
    def reference(self) -> GoldenReference:
        """Return the active golden reference."""
        return self._reference

    def align(self, frame: RectifiedFrame) -> LocalizedPart:
        """Localize ``frame`` against the configured reference.

        Raises:
            FaultError: If the frame cannot be aligned to this reference.
        """
        if frame.camera_id != self._reference.camera_id:
            msg = (
                f"Reference camera {self._reference.camera_id!r} does not match "
                f"frame camera {frame.camera_id!r}"
            )
            raise FaultError(msg)
        height, width = frame.image.shape[:2]
        if (
            width != self._reference.image_width
            or height != self._reference.image_height
        ):
            msg = (
                f"Frame dimensions {width}x{height} do not match reference "
                f"{self._reference.image_width}x{self._reference.image_height}"
            )
            raise FaultError(msg)
        score = 1.0
        if score < self._reference.min_score:
            msg = f"Alignment score {score:.3f} below minimum {self._reference.min_score:.3f}"
            raise FaultError(msg)
        return LocalizedPart(
            frame=frame,
            pose=self._reference.nominal_pose,
            reference_id=self._reference.reference_id,
            reference_ver=self._reference.version,
            score=score,
        )
