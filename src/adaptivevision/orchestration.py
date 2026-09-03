"""Orchestration: the state machine, pipeline, scheduler, and resilience
around one inspection cycle.

The station lifecycle is a finite state machine over
:class:`~adaptivevision.common.StationState`. The pipeline drives one
inspection cycle -- acquire, preprocess, rectify, align, inspect, decide --
with every stage injected as a callable so orchestration stays decoupled
from concrete implementations. The scheduler drives the pipeline through
many cycles. The watchdog, result buffer, and failure handler keep a slow or
failing persistence layer from ever blocking or losing an inspection: a
cycle that exceeds its timeout is a recoverable fault (degrade to REVIEW,
signal pause), and a result that fails to persist gets buffered for later
retry rather than dropped.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from adaptivevision.camera import LocalizedPart
from adaptivevision.common import (
    AnomalyDetector,
    AnomalyResult,
    CameraDriver,
    FaultError,
    Inspector,
    InspectionResult,
    MetrologyResult,
    PartialResult,
    RawFrame,
    RectifiedFrame,
    StationState,
    Verdict,
)
from adaptivevision.config import Recipe
from adaptivevision.decision import DecisionPolicy

# =============================================================================
# Station state machine
#
# Owns the transition table and enforces it: an invalid transition raises
# FaultError, which is non-recoverable and drives the station to a fault /
# safe state. Models the boot path (INIT -> SELF_TEST -> IDLE -> READY ->
# RUNNING) and the fault / shutdown paths.
# =============================================================================

#: Allowed transitions, keyed by source state.
_TRANSITIONS: dict[StationState, frozenset[StationState]] = {
    StationState.INIT: frozenset(
        {StationState.SELF_TEST, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.SELF_TEST: frozenset(
        {StationState.IDLE, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.IDLE: frozenset(
        {
            StationState.READY,
            StationState.CALIBRATION,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.CALIBRATION: frozenset(
        {StationState.IDLE, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.RECIPE_LOADING: frozenset(
        {StationState.READY, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.READY: frozenset(
        {
            StationState.RUNNING,
            StationState.IDLE,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.RUNNING: frozenset(
        {
            StationState.READY,
            StationState.PAUSED,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.PAUSED: frozenset(
        {
            StationState.RUNNING,
            StationState.READY,
            StationState.FAULT,
            StationState.SHUTDOWN,
        }
    ),
    StationState.FAULT: frozenset(
        {StationState.MAINTENANCE, StationState.ESTOP, StationState.SHUTDOWN}
    ),
    StationState.ESTOP: frozenset({StationState.SHUTDOWN}),
    StationState.MAINTENANCE: frozenset(
        {StationState.IDLE, StationState.FAULT, StationState.SHUTDOWN}
    ),
    StationState.SHUTDOWN: frozenset(),
}


class StationStateMachine:
    """A guarded finite state machine over :class:`~adaptivevision.common.StationState`.

    Args:
        initial: The initial state. Defaults to :attr:`StationState.INIT`.
    """

    def __init__(self, initial: StationState = StationState.INIT) -> None:
        """Initialize the state machine."""
        self._state = initial

    @property
    def state(self) -> StationState:
        """Return the current state."""
        return self._state

    def can_transition(self, target: StationState) -> bool:
        """Return ``True`` if a transition to ``target`` is allowed."""
        return target in _TRANSITIONS[self._state]

    def transition(self, target: StationState) -> StationState:
        """Transition to ``target``, enforcing the transition table.

        Args:
            target: The state to transition to.

        Returns:
            The new state.

        Raises:
            FaultError: If the transition is not allowed from the current state.
        """
        if not self.can_transition(target):
            msg = f"Invalid station transition: {self._state.value} -> {target.value}"
            raise FaultError(msg)
        self._state = target
        return self._state

    def to_fault(self) -> StationState:
        """Transition to :attr:`StationState.FAULT` if allowed.

        Returns:
            The new state (``FAULT`` if the transition succeeded, otherwise the
            current state unchanged).
        """
        if self.can_transition(StationState.FAULT):
            return self.transition(StationState.FAULT)
        return self._state


# =============================================================================
# The inspection pipeline
#
# The heart of the walking skeleton: drives one inspection cycle by
# acquiring a frame from the camera driver and producing an
# InspectionResult. Preprocessing, rectification, and alignment are all
# optional stages injected as callables so orchestration stays decoupled
# from concrete implementations.
# =============================================================================

Preprocessor = Callable[[RawFrame], RawFrame]
Rectifier = Callable[[RawFrame], RectifiedFrame]
Aligner = Callable[[RectifiedFrame], LocalizedPart]


def _defects_of(partial: PartialResult) -> tuple[object, ...]:
    """Return the defects carried by a partial result, if any."""
    return getattr(partial, "defects", ())


def new_inspection_id() -> str:
    """Return a unique inspection identifier."""
    return f"inspection-{uuid.uuid4().hex[:12]}"


class InspectionPipeline:
    """Runs a single inspection cycle against a camera driver.

    Args:
        camera: The camera driver to acquire frames from.
        station_id: Identifier of the owning station.
        recipe_ver: Version of the active recipe, for traceability.
    """

    def __init__(
        self,
        camera: CameraDriver,
        *,
        station_id: str,
        recipe_ver: str,
        preprocessor: Preprocessor | None = None,
        rectifier: Rectifier | None = None,
        aligner: Aligner | None = None,
        recipe: Recipe | None = None,
        metrology_inspector: Inspector[LocalizedPart, Recipe] | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        decision_policy: DecisionPolicy | None = None,
    ) -> None:
        """Initialize the pipeline."""
        self._camera = camera
        self._station_id = station_id
        self._recipe_ver = recipe_ver
        self._preprocessor = preprocessor
        self._rectifier = rectifier
        self._aligner = aligner
        self._recipe = recipe
        self._metrology_inspector = metrology_inspector
        self._anomaly_detector = anomaly_detector
        self._decision_policy = decision_policy

    def run(self, part_id: str, *, trigger_id: str | None = None) -> InspectionResult:
        """Execute one inspection cycle.

        Args:
            part_id: Identifier of the part being inspected.
            trigger_id: Identifier of the triggering event, if any.

        Returns:
            The inspection result for the part.

        Raises:
            AcquisitionError: If the frame cannot be acquired.
        """
        started = time.monotonic()
        frame = self._acquire(trigger_id)
        frame = self._preprocess(frame)
        rectified = self._rectify(frame)
        part = self._align(rectified)
        metrology = self._inspect_metrology(part)
        anomaly = self._inspect_anomaly(rectified)
        cycle_time_ms = (time.monotonic() - started) * 1000.0
        measurements = metrology.measurements if metrology is not None else ()
        partials = [p for p in (metrology, anomaly) if p is not None]
        verdict = self._decide(partials)

        return InspectionResult(
            inspection_id=new_inspection_id(),
            part_id=part_id,
            station_id=self._station_id,
            verdict=verdict,
            recipe_ver=self._recipe_ver,
            model_ver="",
            calib_ver=rectified.calibration_ver,
            cycle_time_ms=cycle_time_ms,
            timestamp_utc=datetime.now(UTC),
            measurements=measurements,
            defects=tuple(d for p in partials for d in p.defects),
            anomaly_score=anomaly.score if anomaly is not None else None,
            image_refs=(rectified.frame_id,),
        )

    def _acquire(self, trigger_id: str | None) -> RawFrame:
        """Acquire a frame from the camera driver."""
        return self._camera.capture(trigger_id)

    def _preprocess(self, frame: RawFrame) -> RawFrame:
        """Apply the optional preprocessing stage."""
        if self._preprocessor is None:
            return frame
        return self._preprocessor(frame)

    def _rectify(self, frame: RawFrame) -> RectifiedFrame:
        """Apply the optional rectification stage."""
        if self._rectifier is None:
            return RectifiedFrame(
                image=frame.image,
                camera_id=frame.camera_id,
                frame_id=frame.frame_id,
                calibration_ver="",
                timestamp_monotonic=frame.timestamp_monotonic,
                timestamp_utc=frame.timestamp_utc,
                trigger_id=frame.trigger_id,
            )
        return self._rectifier(frame)

    def _align(self, frame: RectifiedFrame) -> LocalizedPart | None:
        """Apply the optional alignment stage."""
        if self._aligner is None:
            return None
        return self._aligner(frame)

    def _inspect_metrology(self, part: LocalizedPart | None) -> MetrologyResult | None:
        """Apply the optional dimensional metrology stage."""
        if part is None or self._recipe is None or self._metrology_inspector is None:
            return None
        partial = self._metrology_inspector.inspect(part, self._recipe)
        if not isinstance(partial, MetrologyResult):
            msg = "Metrology inspector must return MetrologyResult"
            raise TypeError(msg)
        return partial

    def _inspect_anomaly(self, frame: RectifiedFrame) -> AnomalyResult | None:
        """Apply the optional anomaly-detection stage."""
        if self._anomaly_detector is None:
            return None
        partial = self._anomaly_detector.detect(frame)
        if not isinstance(partial, AnomalyResult):
            msg = "Anomaly detector must return AnomalyResult"
            raise TypeError(msg)
        return partial

    def _decide(self, partials: Sequence[PartialResult]) -> Verdict:
        """Fuse partial results into a verdict using the decision policy."""
        if self._decision_policy is None:
            defects = [d for p in partials for d in _defects_of(p)]
            return Verdict.FAIL if defects else Verdict.PASS
        return self._decision_policy.decide(partials).verdict


# =============================================================================
# The inspection scheduler
#
# Drives the station through inspection cycles while it is in the RUNNING
# state. Deliberately decoupled from the state machine: it only runs cycles
# and returns results. The station controller (app.py) owns the state
# transitions around it.
# =============================================================================


class InspectionScheduler:
    """Runs inspection cycles against a pipeline.

    Args:
        pipeline: The pipeline to drive.
    """

    def __init__(self, pipeline: InspectionPipeline) -> None:
        """Initialize the scheduler."""
        self._pipeline = pipeline

    def run_cycles(
        self,
        part_ids: Iterable[str],
        *,
        on_result: Callable[[InspectionResult], None] | None = None,
    ) -> tuple[InspectionResult, ...]:
        """Run one inspection cycle per part id.

        Args:
            part_ids: Identifiers of the parts to inspect.
            on_result: Optional callback invoked with each result as it is
                produced.

        Returns:
            A tuple of inspection results, one per part id.
        """
        results: list[InspectionResult] = []
        for part_id in part_ids:
            result = self._pipeline.run(part_id)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return tuple(results)


# =============================================================================
# The cycle watchdog
#
# Enforces the maximum allowed inspection cycle time. A cycle that exceeds
# the configured timeout is a recoverable fault: it degrades the part to
# REVIEW and signals the station to pause, rather than halting the process.
# =============================================================================


class CycleWatchdog:
    """Monitors inspection cycle times against a configured timeout.

    Args:
        timeout_ms: Maximum allowed cycle time in milliseconds.
    """

    def __init__(self, timeout_ms: float) -> None:
        """Initialize the watchdog with a timeout."""
        self._timeout_ms = timeout_ms
        self._violations = 0

    @property
    def timeout_ms(self) -> float:
        """Return the configured timeout in milliseconds."""
        return self._timeout_ms

    @property
    def violations(self) -> int:
        """Return the number of timeout violations observed."""
        return self._violations

    def check(self, result: InspectionResult) -> bool:
        """Check a result against the timeout.

        Args:
            result: The inspection result to check.

        Returns:
            ``True`` if the cycle exceeded the timeout (a violation), ``False``
            otherwise.
        """
        if result.cycle_time_ms > self._timeout_ms:
            self._violations += 1
            return True
        return False


# =============================================================================
# In-memory result buffering and failure handling
#
# ResultBuffer holds inspection results that could not be persisted
# immediately so they can be retried later without losing data.
# FailureHandler wraps a persistence callable, retries it a bounded number
# of times, and buffers the result for later retry if it still fails.
# =============================================================================


class ResultBuffer:
    """A bounded FIFO buffer of inspection results awaiting persistence."""

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize an empty buffer with a maximum ``capacity``."""
        if capacity <= 0:
            msg = "ResultBuffer capacity must be positive"
            raise ValueError(msg)
        self._capacity = capacity
        self._items: deque[InspectionResult] = deque()

    def push(self, result: InspectionResult) -> None:
        """Append a result, dropping the oldest if the buffer is full."""
        self._items.append(result)
        if len(self._items) > self._capacity:
            self._items.popleft()

    def drain(self) -> tuple[InspectionResult, ...]:
        """Remove and return all buffered results."""
        items = tuple(self._items)
        self._items.clear()
        return items

    def __len__(self) -> int:
        """Return the number of buffered results."""
        return len(self._items)

    def is_full(self) -> bool:
        """Return ``True`` if the buffer is at capacity."""
        return len(self._items) >= self._capacity

    def extend(self, results: Iterable[InspectionResult]) -> None:
        """Append multiple results."""
        for result in results:
            self.push(result)


@dataclass(frozen=True, slots=True)
class FailureOutcome:
    """Outcome of a persistence attempt.

    Attributes:
        persisted: Whether the result was persisted successfully.
        attempts: Number of attempts made.
        buffered: Whether the result was buffered for a later retry.
    """

    persisted: bool
    attempts: int
    buffered: bool


class FailureHandler:
    """Retries persistence and buffers results that still fail."""

    def __init__(
        self,
        persist: Callable[[InspectionResult], None],
        *,
        max_attempts: int = 3,
        buffer: ResultBuffer | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            persist: Callable that persists a single result.
            max_attempts: Maximum number of attempts before buffering.
            buffer: Buffer for results that could not be persisted.
        """
        if max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)
        self._persist = persist
        self._max_attempts = max_attempts
        self._buffer = buffer if buffer is not None else ResultBuffer()

    def handle(self, result: InspectionResult) -> FailureOutcome:
        """Persist ``result`` with retries, buffering on persistent failure.

        Args:
            result: The inspection result to persist.

        Returns:
            The outcome of the persistence attempt.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._persist(result)
                return FailureOutcome(persisted=True, attempts=attempt, buffered=False)
            except Exception:
                if attempt == self._max_attempts:
                    self._buffer.push(result)
                    return FailureOutcome(persisted=False, attempts=attempt, buffered=True)
        # Unreachable; kept for type-checker completeness.
        return FailureOutcome(persisted=False, attempts=self._max_attempts, buffered=True)

    def flush(self) -> tuple[InspectionResult, ...]:
        """Attempt to persist all buffered results.

        Returns:
            The results that still could not be persisted.
        """
        remaining: list[InspectionResult] = []
        for result in self._buffer.drain():
            outcome = self.handle(result)
            if not outcome.persisted:
                remaining.append(result)
        return tuple(remaining)
