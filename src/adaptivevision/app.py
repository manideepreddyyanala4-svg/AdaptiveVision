"""The composition root: wires every seam to a concrete implementation and
assembles the station.

This is the one file that knows about every other module in the package --
that's deliberate. Everything else only imports the small interface it
personally needs (a seam from ``common.py``, a specific class from
``camera.py``), never the whole system; this file is where those pieces
actually get connected. Nothing else in the codebase constructs these
collaborators directly.

Every optional subsystem (calibration, alignment, an anomaly-detection
model, a recipe) follows the same null-object pattern: when it isn't
configured, the walking skeleton still runs end-to-end without it, just with
that stage skipped. :func:`build_station` is the single function that reads
validated configuration and returns a fully wired, ready-to-boot
:class:`StationController`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from adaptivevision.camera import (
    CalibrationRectifier,
    GoldenReference,
    LocalizedPart,
    NullCameraDriver,
    PreprocessingPipeline,
    PreprocessStep,
    ReferenceAligner,
    ensure_grayscale,
    load_calibration,
    load_golden_reference,
    resize_to,
)
from adaptivevision.common import (
    AdaptiveVisionError,
    AnomalyDetector,
    CameraDriver,
    CameraKind,
    ExecutionProvider,
    InspectionResult,
    RawFrame,
    RectifiedFrame,
    Severity,
    StationState,
)
from adaptivevision.config import CameraConfig, JsonRecipeStore, Recipe, StationConfig
from adaptivevision.decision import DecisionPolicy
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import ThresholdAnomalyDetector
from adaptivevision.orchestration import (
    CycleWatchdog,
    InspectionPipeline,
    InspectionScheduler,
    StationStateMachine,
)
from adaptivevision.storage import make_persistence_handler, open_database
from adaptivevision.storage import SqliteResultRepository

#: Type of the ``on_result`` persistence hook.
OnResult = Callable[[InspectionResult], None]

#: Type of the preprocessing hook injected into the pipeline.
Preprocessor = Callable[[RawFrame], RawFrame]

#: Type of the rectification hook injected into the pipeline.
Rectifier = Callable[[RawFrame], RectifiedFrame]

#: Type of the alignment hook injected into the pipeline.
Aligner = Callable[[RectifiedFrame], LocalizedPart]


# =============================================================================
# The station controller
#
# The composition root's orchestrator: owns the station state machine and
# drives the inspection pipeline through the scheduler, enforcing
# cycle-time limits with the watchdog. It is the single object the
# application entrypoint interacts with.
# =============================================================================


class StationController:
    """Coordinates the station lifecycle and inspection cycles.

    Args:
        state_machine: The station state machine.
        pipeline: The inspection pipeline.
        scheduler: The inspection scheduler.
        watchdog: The cycle watchdog.
        on_result: Optional callback invoked with each result as it is produced
            (used to persist results off the critical path).
    """

    def __init__(
        self,
        state_machine: StationStateMachine,
        pipeline: InspectionPipeline,
        scheduler: InspectionScheduler,
        watchdog: CycleWatchdog,
        on_result: OnResult | None = None,
    ) -> None:
        """Initialize the controller with its collaborators."""
        self._state = state_machine
        self._pipeline = pipeline
        self._scheduler = scheduler
        self._watchdog = watchdog
        self._on_result = on_result

    @property
    def state(self) -> StationState:
        """Return the current station state."""
        return self._state.state

    def boot(self) -> None:
        """Run the boot sequence: ``INIT -> SELF_TEST -> IDLE``.

        Raises:
            FaultError: If a transition is invalid.
        """
        self._state.transition(StationState.SELF_TEST)
        self._state.transition(StationState.IDLE)

    def ready(self) -> None:
        """Transition the station to ``READY``.

        Raises:
            FaultError: If the transition is invalid.
        """
        self._state.transition(StationState.READY)

    def run(self, part_ids: list[str]) -> tuple[InspectionResult, ...]:
        """Inspect a batch of parts.

        Transitions to ``RUNNING``, runs one cycle per part, then returns to
        ``READY``.

        Args:
            part_ids: Identifiers of the parts to inspect.

        Returns:
            A tuple of inspection results, one per part.

        Raises:
            FaultError: If the station is not in a state that can run.
            AdaptiveVisionError: If a cycle fails.
        """
        self._state.transition(StationState.RUNNING)
        try:
            results = self._scheduler.run_cycles(part_ids, on_result=self._on_result)
        except AdaptiveVisionError:
            self._state.to_fault()
            raise
        finally:
            if self._state.state is StationState.RUNNING:
                self._state.transition(StationState.READY)
        return results

    def shutdown(self) -> None:
        """Transition the station to ``SHUTDOWN``.

        Raises:
            FaultError: If the transition is invalid.
        """
        self._state.transition(StationState.SHUTDOWN)


# =============================================================================
# Builders
#
# One function per optional subsystem, each following the same null-object
# pattern: absent configuration means None/a synthetic default, not an
# error, so the walking skeleton always runs end-to-end.
# =============================================================================


def build_camera(config: StationConfig) -> CameraDriver:
    """Build the camera driver for the station.

    Uses the null-object strategy: when no camera is configured, a synthetic
    :class:`~adaptivevision.camera.NullCameraDriver` is returned so the
    walking skeleton runs without hardware.

    Args:
        config: The validated station configuration.

    Returns:
        A :class:`~adaptivevision.common.CameraDriver` ready to be opened.
    """
    if not config.cameras:
        # No camera configured: use a synthetic 640x480 null-object driver.
        synthetic = CameraConfig(
            camera_id="null",
            kind=CameraKind.AREA_SCAN_2D,
            width=640,
            height=480,
            fps=30.0,
        )
        return NullCameraDriver(synthetic)

    camera_id = next(iter(config.cameras))
    return NullCameraDriver(config.camera(camera_id))


def build_persistence(config: StationConfig) -> tuple[SqliteResultRepository, OnResult]:
    """Build the local persistence layer for the station.

    Args:
        config: The validated station configuration.

    Returns:
        A tuple of ``(repository, on_result_handler)`` where the handler is a
        callable suitable for the station's ``on_result`` hook.
    """
    db_path = config.extra.get("DB_PATH", "adaptivevision.db")
    _, session_factory = open_database(db_path)
    repository = SqliteResultRepository(session_factory)
    handler = make_persistence_handler(repository)
    return repository, handler


def build_preprocessor(config: StationConfig) -> Preprocessor:
    """Build the deterministic preprocessing stage.

    Args:
        config: The validated station configuration.

    Returns:
        A callable that preprocesses raw frames before calibration. Resizes
        to ``MODEL_INPUT_HEIGHT``/``MODEL_INPUT_WIDTH`` when both are set, so
        a configured anomaly-detection model (see :func:`build_anomaly_detector`)
        always receives frames matching its fixed input contract.
    """
    steps: list[PreprocessStep] = []
    if config.extra.get("PREPROCESS_GRAYSCALE", True) is not False:
        steps.append(ensure_grayscale)
    height = config.extra.get("MODEL_INPUT_HEIGHT")
    width = config.extra.get("MODEL_INPUT_WIDTH")
    if height is not None and width is not None:
        steps.append(resize_to(int(height), int(width)))
    return PreprocessingPipeline(tuple(steps)).apply


def build_rectifier(config: StationConfig) -> Rectifier | None:
    """Build an optional calibration rectifier from configuration.

    Args:
        config: The validated station configuration.

    Returns:
        A rectification callable when ``CALIBRATION_PATH`` is configured,
        otherwise ``None`` so uncalibrated skeleton runs remain possible.
    """
    calibration_path = config.extra.get("CALIBRATION_PATH")
    if calibration_path is None:
        return None
    calibration = load_calibration(str(calibration_path))
    return CalibrationRectifier(calibration).apply


def build_aligner(config: StationConfig) -> Aligner | None:
    """Build an optional golden-reference aligner from configuration.

    Args:
        config: The validated station configuration.

    Returns:
        An alignment callable when ``REFERENCE_PATH`` is configured, otherwise
        ``None`` so skeleton runs remain possible.
    """
    reference_path = config.extra.get("REFERENCE_PATH")
    if reference_path is None:
        return None
    reference: GoldenReference = load_golden_reference(str(reference_path))
    return ReferenceAligner(reference).align


def build_recipe(config: StationConfig) -> Recipe | None:
    """Load the optional active recipe from configuration.

    Args:
        config: The validated station configuration.

    Returns:
        The recipe named by ``default_recipe_id``, loaded from the directory
        named by ``RECIPE_DIR`` (default ``recipes``), or ``None`` when no
        recipe is configured -- skeleton runs remain possible without one.

    Raises:
        RecipeError: If a recipe *is* configured but missing or invalid; this
            is a fault condition, not something to silently degrade from.
    """
    if config.default_recipe_id is None:
        return None
    recipe_dir = Path(str(config.extra.get("RECIPE_DIR", "recipes")))
    return JsonRecipeStore(recipe_dir).load(config.default_recipe_id)


def _resolve_providers(config: StationConfig) -> tuple[ExecutionProvider, ...]:
    """Resolve ``config.execution_provider`` to an ONNX Runtime provider list.

    Args:
        config: The validated station configuration.

    Returns:
        The configured provider first, falling back to CPU so a station
        configured for e.g. TensorRT still runs on edge hardware where that
        provider isn't available -- ONNX Runtime tries providers in order and
        falls back through the list itself. CPU alone if that's the
        configured provider.
    """
    provider = config.execution_provider
    if provider is ExecutionProvider.CPU:
        return (ExecutionProvider.CPU,)
    return (provider, ExecutionProvider.CPU)


def build_anomaly_detector(
    config: StationConfig,
    recipe: Recipe | None,
) -> AnomalyDetector | None:
    """Build an optional anomaly detector from configuration.

    Wired only when ``MODEL_PATH`` names an ONNX model to load, following the
    same optional-hardware pattern as calibration/alignment: skeleton runs
    without a model configured remain possible. The threshold and the
    severity assigned to an anomaly (which the decision policy turns into
    FAIL or REVIEW) come from the active recipe's declared decision policy
    when one is loaded, otherwise from safe defaults.

    Args:
        config: The validated station configuration.
        recipe: The active recipe, if any (see :func:`build_recipe`).

    Returns:
        A :class:`~adaptivevision.metrology.ThresholdAnomalyDetector` backed
        by ONNX Runtime, or ``None`` when no model is configured.
    """
    model_path = config.extra.get("MODEL_PATH")
    if model_path is None:
        return None

    engine = OnnxInferenceEngine(
        model_dir=str(config.extra.get("MODEL_DIR", "models")),
        providers=_resolve_providers(config),
    )
    engine.load(str(model_path))
    engine.warmup()

    threshold = recipe.decision.anomaly_threshold if recipe is not None else 0.5
    anomalous_severity = (
        Severity.MINOR
        if recipe is not None and recipe.decision.review_on_anomaly
        else Severity.MAJOR
    )
    return ThresholdAnomalyDetector(
        engine,
        threshold,
        anomalous_severity=anomalous_severity,
    )


def build_decision_policy(recipe: Recipe | None) -> DecisionPolicy | None:
    """Build the live decision policy from the active recipe, if any.

    Args:
        recipe: The active recipe, if any (see :func:`build_recipe`).

    Returns:
        A policy translated from the recipe's declared decision contract via
        :meth:`~adaptivevision.decision.DecisionPolicy.from_recipe`, or
        ``None`` when no recipe is configured -- the pipeline then falls back
        to its own any-defect-fails rule.
    """
    if recipe is None:
        return None
    return DecisionPolicy.from_recipe(recipe)


def build_station(config: StationConfig) -> StationController:
    """Assemble the full station from validated configuration.

    Args:
        config: The validated station configuration.

    Returns:
        A fully wired :class:`StationController`.
    """
    camera = build_camera(config)
    camera.open()

    recipe = build_recipe(config)

    pipeline = InspectionPipeline(
        camera,
        station_id=config.station_id,
        recipe_ver=config.default_recipe_id or "unset",
        preprocessor=build_preprocessor(config),
        rectifier=build_rectifier(config),
        aligner=build_aligner(config),
        recipe=recipe,
        anomaly_detector=build_anomaly_detector(config, recipe),
        decision_policy=build_decision_policy(recipe),
    )
    scheduler = InspectionScheduler(pipeline)
    watchdog = CycleWatchdog(config.cycle_timeout_ms)
    state_machine = StationStateMachine()

    _, on_result = build_persistence(config)

    return StationController(
        state_machine=state_machine,
        pipeline=pipeline,
        scheduler=scheduler,
        watchdog=watchdog,
        on_result=on_result,
    )
