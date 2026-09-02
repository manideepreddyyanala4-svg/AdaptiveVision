"""Capture stage: get a usable image of the part.

Covers the full path from a raw camera frame to a rectified, aligned,
model-ready image: the camera driver seam and its null-object implementation,
the threaded frame buffer, lens/optical calibration and its hot-swap
lifecycle, golden-reference alignment, and deterministic preprocessing.

The walking skeleton must run end-to-end without physical hardware. Per the
frozen null-object strategy, every seam has a no-op / synthetic
implementation injected when no real adapter is configured:
:class:`NullCameraDriver` is always "healthy," opens and closes without side
effects, and produces a synthetic grayscale frame of the configured size.
Real camera backends replace it behind the same :class:`~adaptivevision.common.CameraDriver`
seam without changing anything downstream.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeAlias

import numpy as np

from adaptivevision.common import (
    CalibrationError,
    CameraDriver,
    FaultError,
    Pose,
    RawFrame,
    RectifiedFrame,
)
from adaptivevision.config import CameraConfig

if TYPE_CHECKING:
    from adaptivevision.common import Image

# =============================================================================
# Frame construction
# =============================================================================


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


# =============================================================================
# Camera drivers
# =============================================================================


class NullCameraDriver(CameraDriver):
    """A synthetic :class:`~adaptivevision.common.CameraDriver` used when no
    real camera is configured.

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
            from adaptivevision.common import AcquisitionError

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


class ThreadedFrameBuffer:
    """Runs a :class:`~adaptivevision.common.CameraDriver` on a dedicated
    background thread (Milestone M21).

    :class:`~adaptivevision.common.CameraDriver` is deliberately a blocking,
    single-threaded seam ("a single acquisition thread owns the driver") --
    the right default for the walking skeleton, but not for a line where a
    slow or stalled capture call must never stall whatever consumes frames
    next (model inference). This wraps any driver, by composition rather than
    by changing the seam, in a producer/consumer buffer: a background thread
    continuously calls ``driver.capture()`` and pushes into a bounded queue,
    while :meth:`get_latest_frame` always returns immediately with the newest
    frame available.

    Overflow policy: when the buffer is full, capturing a new frame discards
    the oldest buffered one to make room. The capture thread therefore never
    blocks waiting for a slow consumer, and :meth:`get_latest_frame` always
    hands back the most recent frame the camera has produced rather than
    working through a backlog of stale ones -- the right trade on a live
    line, where an old frame is not worth inspecting.

    A capture failure (any exception from ``driver.capture()``, not only
    :class:`~adaptivevision.common.AcquisitionError`) is recorded via
    :meth:`last_error` and the loop keeps running -- a background thread that
    dies silently on the first unexpected error would be a worse failure mode
    than a slightly-broad catch here; Python does not propagate a thread's
    exception to its caller on its own.

    Args:
        driver: The camera driver to wrap. The caller owns its ``open()``/
            ``close()`` lifecycle (call ``open()`` before :meth:`start`),
            matching every other use of :class:`~adaptivevision.common.CameraDriver`
            in this codebase (see ``app.py::build_camera``).
        maxsize: Bounded buffer depth.
        poll_interval_s: Optional delay between capture attempts. ``None``
            (the default) captures as fast as the driver allows -- fine for
            a real camera whose ``capture()`` call itself takes real time,
            wasteful for a driver like :class:`NullCameraDriver` that
            returns instantly; pass e.g. ``1.0 / camera_config.fps`` to pace
            it explicitly.
    """

    def __init__(
        self,
        driver: CameraDriver,
        *,
        maxsize: int = 10,
        poll_interval_s: float | None = None,
    ) -> None:
        """Initialize the buffer without starting the capture thread."""
        self._driver = driver
        self._poll_interval_s = poll_interval_s
        self._queue: queue.Queue[RawFrame] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error_lock = threading.Lock()
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        """Whether the background capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background capture thread. A no-op if already running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ThreadedFrameBuffer", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float | None = 2.0) -> None:
        """Signal the capture thread to stop and wait for it to exit.

        Args:
            timeout: Maximum time to wait for the thread to exit.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_latest_frame(self) -> RawFrame | None:
        """Return the most recently captured frame without blocking.

        Returns:
            The newest buffered frame, or ``None`` if none has been captured
            yet (or none survived the overflow policy since the last read).
        """
        latest: RawFrame | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                return latest

    def last_error(self) -> Exception | None:
        """Return the most recent capture failure, without clearing it."""
        with self._error_lock:
            return self._last_error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._driver.capture()
            except Exception as exc:  # noqa: BLE001 -- see class docstring
                with self._error_lock:
                    self._last_error = exc
            else:
                self._push(frame)
            if self._poll_interval_s is not None:
                time.sleep(self._poll_interval_s)

    def _push(self, frame: RawFrame) -> None:
        """Enqueue ``frame``, dropping the oldest buffered frame if full."""
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass  # a concurrent reader already drained it; retry the put


# =============================================================================
# Calibration: artifact model, loading, rectification, hot-swap lifecycle
# =============================================================================


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Versioned calibration for a single camera.

    Attributes:
        calibration_id: Stable identifier of the calibration artifact.
        version: Version string recorded into inspection lineage.
        camera_id: Camera this calibration applies to.
        image_width: Calibrated image width in pixels.
        image_height: Calibrated image height in pixels.
        pixel_size_mm: Physical size represented by one pixel.
        intrinsic_matrix: 3x3 camera matrix.
        distortion_coefficients: Lens distortion coefficients.
    """

    calibration_id: str
    version: str
    camera_id: str
    image_width: int
    image_height: int
    pixel_size_mm: float
    intrinsic_matrix: tuple[tuple[float, float, float], ...]
    distortion_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate calibration invariants."""
        if not self.calibration_id:
            msg = "CameraCalibration.calibration_id must not be empty"
            raise CalibrationError(msg)
        if not self.version:
            msg = "CameraCalibration.version must not be empty"
            raise CalibrationError(msg)
        if not self.camera_id:
            msg = "CameraCalibration.camera_id must not be empty"
            raise CalibrationError(msg)
        if self.image_width <= 0 or self.image_height <= 0:
            msg = "CameraCalibration image dimensions must be positive"
            raise CalibrationError(msg)
        if self.pixel_size_mm <= 0.0:
            msg = "CameraCalibration.pixel_size_mm must be positive"
            raise CalibrationError(msg)
        if len(self.intrinsic_matrix) != 3 or any(
            len(row) != 3 for row in self.intrinsic_matrix
        ):
            msg = "CameraCalibration.intrinsic_matrix must be 3x3"
            raise CalibrationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "calibration_id": self.calibration_id,
            "version": self.version,
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "pixel_size_mm": self.pixel_size_mm,
            "intrinsic_matrix": [list(row) for row in self.intrinsic_matrix],
            "distortion_coefficients": list(self.distortion_coefficients),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize and validate a calibration artifact."""
        return cls(
            calibration_id=data["calibration_id"],
            version=data["version"],
            camera_id=data["camera_id"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            pixel_size_mm=data["pixel_size_mm"],
            intrinsic_matrix=_matrix_from_raw(data["intrinsic_matrix"]),
            distortion_coefficients=tuple(
                float(v) for v in data.get("distortion_coefficients", ())
            ),
        )


def identity_calibration(
    *,
    camera_id: str,
    width: int,
    height: int,
    version: str = "identity",
    pixel_size_mm: float = 1.0,
) -> CameraCalibration:
    """Build an explicit identity calibration for synthetic or uncalibrated runs."""
    return CameraCalibration(
        calibration_id=f"{camera_id}-{version}",
        version=version,
        camera_id=camera_id,
        image_width=width,
        image_height=height,
        pixel_size_mm=pixel_size_mm,
        intrinsic_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _matrix_from_raw(raw: Any) -> tuple[tuple[float, float, float], ...]:
    """Convert a raw JSON matrix into a typed 3x3 tuple."""
    rows = tuple(tuple(float(v) for v in row) for row in raw)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        msg = "CameraCalibration.intrinsic_matrix must be 3x3"
        raise CalibrationError(msg)
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )


def load_calibration(path: str | Path) -> CameraCalibration:
    """Load a calibration artifact from JSON.

    Args:
        path: Filesystem path to the calibration JSON document.

    Returns:
        The validated calibration artifact.

    Raises:
        CalibrationError: If the file cannot be read, parsed, or validated.
    """
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to load calibration artifact {artifact_path}: {exc}"
        raise CalibrationError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Calibration artifact {artifact_path} must contain a JSON object"
        raise CalibrationError(msg)
    try:
        return CameraCalibration.from_dict(payload)
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        msg = f"Invalid calibration artifact {artifact_path}: {exc}"
        raise CalibrationError(msg) from exc


class CalibrationRectifier:
    """Apply a loaded calibration artifact to raw frames.

    Preserves deterministic lineage and validates that frames match the
    loaded calibration. Optical undistortion maps can replace the current
    identity image transform later without changing the pipeline contract.
    """

    def __init__(self, calibration: CameraCalibration) -> None:
        """Initialize with a validated calibration artifact."""
        self._calibration = calibration

    @property
    def calibration(self) -> CameraCalibration:
        """Return the active calibration artifact."""
        return self._calibration

    def apply(self, frame: RawFrame) -> RectifiedFrame:
        """Rectify ``frame`` and attach calibration lineage.

        Raises:
            CalibrationError: If the frame belongs to another camera or has
                dimensions different from the calibration artifact.
        """
        if frame.camera_id != self._calibration.camera_id:
            msg = (
                f"Calibration camera {self._calibration.camera_id!r} does not match "
                f"frame camera {frame.camera_id!r}"
            )
            raise CalibrationError(msg)
        height, width = frame.image.shape[:2]
        if width != self._calibration.image_width or height != self._calibration.image_height:
            msg = (
                f"Frame dimensions {width}x{height} do not match calibration "
                f"{self._calibration.image_width}x{self._calibration.image_height}"
            )
            raise CalibrationError(msg)
        return RectifiedFrame(
            image=frame.image.copy(),
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            calibration_ver=self._calibration.version,
            timestamp_monotonic=frame.timestamp_monotonic,
            timestamp_utc=frame.timestamp_utc,
            trigger_id=frame.trigger_id,
        )


@dataclass(frozen=True, slots=True)
class SelfTestResult:
    """Outcome of a calibration self-test.

    Attributes:
        passed: Whether the calibration passed all checks.
        checks: Human-readable results of each performed check.
    """

    passed: bool
    checks: tuple[str, ...]


class CalibrationSelfTest:
    """Runs validation checks against a calibration artifact."""

    def run(self, calibration: CameraCalibration) -> SelfTestResult:
        """Validate ``calibration`` and return the self-test outcome.

        Args:
            calibration: The calibration artifact to validate.

        Returns:
            The self-test result.
        """
        checks: list[str] = []
        passed = True

        if calibration.pixel_size_mm > 0.0:
            checks.append("pixel_size_mm positive")
        else:
            checks.append("pixel_size_mm NOT positive")
            passed = False

        if self._matrix_is_invertible(calibration.intrinsic_matrix):
            checks.append("intrinsic_matrix invertible")
        else:
            checks.append("intrinsic_matrix NOT invertible")
            passed = False

        if calibration.image_width > 0 and calibration.image_height > 0:
            checks.append("image dimensions positive")
        else:
            checks.append("image dimensions NOT positive")
            passed = False

        return SelfTestResult(passed=passed, checks=tuple(checks))

    @staticmethod
    def _matrix_is_invertible(
        matrix: tuple[tuple[float, float, float], ...],
    ) -> bool:
        """Return ``True`` if the 3x3 matrix has a non-zero determinant."""
        (a, b, c), (d, e, f), (g, h, i) = matrix
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        return abs(determinant) > 1e-12


class CalibrationManager:
    """Manages the active calibration per camera with hot-swap support.

    The manager keeps the currently active calibration for each camera and
    supports atomic replacement. A new calibration is validated by the
    self-test before it becomes active.
    """

    def __init__(self, self_test: CalibrationSelfTest | None = None) -> None:
        """Initialize an empty manager."""
        self._self_test = self_test or CalibrationSelfTest()
        self._active: dict[str, CameraCalibration] = {}

    def active(self, camera_id: str) -> CameraCalibration | None:
        """Return the active calibration for a camera, or ``None``."""
        return self._active.get(camera_id)

    def activate(self, calibration: CameraCalibration) -> None:
        """Hot-swap ``calibration`` as the active artifact for its camera.

        Args:
            calibration: The calibration to activate.

        Raises:
            CalibrationError: If the calibration fails its self-test.
        """
        result = self._self_test.run(calibration)
        if not result.passed:
            msg = f"Calibration {calibration.calibration_id} failed self-test"
            raise CalibrationError(msg)
        self._active[calibration.camera_id] = calibration

    def cameras(self) -> tuple[str, ...]:
        """Return the camera ids with an active calibration."""
        return tuple(sorted(self._active))


# =============================================================================
# Alignment: golden-reference model, loading, and localization
# =============================================================================


@dataclass(frozen=True, slots=True)
class GoldenReference:
    """Versioned 2D reference used to localize a part.

    Attributes:
        reference_id: Stable identifier of the golden reference.
        version: Version string of this reference artifact.
        camera_id: Camera this reference applies to.
        image_width: Expected rectified image width.
        image_height: Expected rectified image height.
        nominal_pose: Pose of a correctly aligned part in reference space.
        min_score: Minimum alignment score accepted by the aligner.
    """

    reference_id: str
    version: str
    camera_id: str
    image_width: int
    image_height: int
    nominal_pose: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0))
    min_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate reference invariants."""
        if not self.reference_id:
            msg = "GoldenReference.reference_id must not be empty"
            raise FaultError(msg)
        if not self.version:
            msg = "GoldenReference.version must not be empty"
            raise FaultError(msg)
        if not self.camera_id:
            msg = "GoldenReference.camera_id must not be empty"
            raise FaultError(msg)
        if self.image_width <= 0 or self.image_height <= 0:
            msg = "GoldenReference image dimensions must be positive"
            raise FaultError(msg)
        if not 0.0 <= self.min_score <= 1.0:
            msg = "GoldenReference.min_score must be in [0, 1]"
            raise FaultError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "reference_id": self.reference_id,
            "version": self.version,
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "nominal_pose": self.nominal_pose.to_dict(),
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            reference_id=data["reference_id"],
            version=data["version"],
            camera_id=data["camera_id"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            nominal_pose=Pose.from_dict(
                data.get("nominal_pose", {"x": 0.0, "y": 0.0, "theta_deg": 0.0})
            ),
            min_score=data.get("min_score", 0.0),
        )


@dataclass(frozen=True, slots=True)
class LocalizedPart:
    """A rectified frame localized against a golden reference."""

    frame: RectifiedFrame
    pose: Pose
    reference_id: str
    reference_ver: str
    score: float

    def __post_init__(self) -> None:
        """Validate localized-part invariants."""
        if not 0.0 <= self.score <= 1.0:
            msg = "LocalizedPart.score must be in [0, 1]"
            raise FaultError(msg)

    def lineage(self) -> dict[str, Any]:
        """Return JSON-friendly alignment lineage."""
        return {
            "frame_id": self.frame.frame_id,
            "calibration_ver": self.frame.calibration_ver,
            "reference_id": self.reference_id,
            "reference_ver": self.reference_ver,
            "pose": self.pose.to_dict(),
            "score": self.score,
        }


def load_golden_reference(path: str | Path) -> GoldenReference:
    """Load a golden-reference artifact from JSON.

    Args:
        path: Filesystem path to the reference JSON document.

    Returns:
        The validated golden reference.

    Raises:
        FaultError: If the artifact cannot be read, parsed, or validated.
    """
    reference_path = Path(path)
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to load golden reference {reference_path}: {exc}"
        raise FaultError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Golden reference {reference_path} must contain a JSON object"
        raise FaultError(msg)
    try:
        return GoldenReference.from_dict(payload)
    except (KeyError, TypeError, ValueError, FaultError) as exc:
        msg = f"Invalid golden reference {reference_path}: {exc}"
        raise FaultError(msg) from exc


class ReferenceAligner:
    """Localize a rectified frame against a versioned golden reference.

    The default estimator is deterministic and conservative: it validates
    that the frame matches the reference camera and dimensions, then emits
    the reference's nominal pose with a perfect score. Rich feature/template
    matching can replace this estimator behind the same callable boundary
    without changing downstream contracts.
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
        if width != self._reference.image_width or height != self._reference.image_height:
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


# =============================================================================
# Preprocessing: deterministic operators applied before inference
# =============================================================================

PreprocessStep: TypeAlias = Callable[[RawFrame], RawFrame]


def normalize_uint8(frame: RawFrame) -> RawFrame:
    """Scale an image to the full uint8 range without mutating the input frame."""
    image = frame.image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value == min_value:
        normalized = np.zeros_like(frame.image, dtype=np.uint8)
    else:
        normalized = ((image - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)
    return _replace_image(frame, normalized)


def ensure_grayscale(frame: RawFrame) -> RawFrame:
    """Convert RGB/RGBA images to grayscale; grayscale inputs pass through as copies."""
    image = frame.image
    if image.ndim == 2:
        return _replace_image(frame, image.copy())
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        msg = "Expected a grayscale, RGB, or RGBA image"
        raise ValueError(msg)
    rgb = image[:, :, :3].astype(np.float32)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(image.dtype)
    return _replace_image(frame, gray)


def resize_to(height: int, width: int) -> PreprocessStep:
    """Build a step that resizes a frame to a model's fixed ``(height, width)``.

    Uses nearest-neighbor sampling -- adequate for matching an inference
    contract, and dependency-free (no OpenCV in the production package; that
    stays a training-only dependency).

    Args:
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        A preprocessing step producing a ``(height, width[, channels])`` image.
    """

    def _resize(frame: RawFrame) -> RawFrame:
        image = frame.image
        src_height, src_width = image.shape[0], image.shape[1]
        row_idx = (np.arange(height) * src_height / height).astype(np.intp)
        col_idx = (np.arange(width) * src_width / width).astype(np.intp)
        resized = image[row_idx][:, col_idx]
        return _replace_image(frame, resized)

    return _resize


class PreprocessingPipeline:
    """Apply preprocessing steps in order to a raw frame."""

    def __init__(self, steps: tuple[PreprocessStep, ...] = ()) -> None:
        """Initialize the pipeline."""
        self._steps = steps

    def apply(self, frame: RawFrame) -> RawFrame:
        """Apply all configured preprocessing steps."""
        current = frame
        for step in self._steps:
            current = step(current)
        return current


def _replace_image(frame: RawFrame, image: np.ndarray[Any, np.dtype[Any]]) -> RawFrame:
    """Return ``frame`` metadata with a replacement image."""
    return RawFrame(
        image=image,
        camera_id=frame.camera_id,
        frame_id=frame.frame_id,
        timestamp_monotonic=frame.timestamp_monotonic,
        timestamp_utc=frame.timestamp_utc,
        trigger_id=frame.trigger_id,
    )
