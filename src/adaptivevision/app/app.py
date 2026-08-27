"""The application composition root (Milestone M3, extended M4/M5/M6).

The composition root is the only place that wires concrete implementations to
the abstraction seams (Architecture Spec v1.0 §19). It reads the validated
:class:`~adaptivevision.config.settings.StationConfig`, constructs the camera
driver (using the null-object strategy when no real camera is configured), and
assembles the orchestration layer into a :class:`StationController`.

Milestone M4 extends the composition root with the local persistence layer: the
SQLite database, the result repository, and a persistence handler that is wired
into the station's ``on_result`` hook so results are persisted off the
inspection critical path.

Milestone M5 adds optional preprocessing and calibration rectification wiring.
The concrete loaders/operators are still created only here and injected into
the pipeline as callables.

Milestone M6 adds optional golden-reference alignment wiring.

M9/M10 wiring (previously declared but never connected to this composition
root -- see docs/milestones/M20.md): when ``MODEL_PATH`` is configured, a real
ONNX-backed anomaly detector is loaded and injected; when a recipe is
configured, its declared decision policy is translated into a live
:class:`~adaptivevision.decision.DecisionPolicy` and injected too. Without
either, the pipeline still runs (skeleton-without-hardware stays true) but
every part is unconditionally PASS, exactly as before -- the difference is
that a real model now produces a real verdict once one is configured, instead
of that being permanently unreachable regardless of configuration.

Nothing else in the codebase constructs these collaborators directly; they are
injected here and passed down.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from adaptivevision.acquisition.camera import NullCameraDriver
from adaptivevision.alignment import (
    LocalizedPart,
    ReferenceAligner,
    load_golden_reference,
)
from adaptivevision.app.station import StationController
from adaptivevision.calibration import CalibrationRectifier, load_calibration
from adaptivevision.common.enums import ExecutionProvider, Severity
from adaptivevision.common.interfaces import AnomalyDetector, CameraDriver
from adaptivevision.common.result import InspectionResult
from adaptivevision.common.types import RawFrame, RectifiedFrame
from adaptivevision.config.settings import StationConfig
from adaptivevision.decision import DecisionPolicy
from adaptivevision.inference.onnx import OnnxInferenceEngine
from adaptivevision.inspection.anomaly.detector import ThresholdAnomalyDetector
from adaptivevision.orchestration.pipeline import InspectionPipeline
from adaptivevision.orchestration.scheduler import InspectionScheduler
from adaptivevision.orchestration.state import StationStateMachine
from adaptivevision.orchestration.watchdog import CycleWatchdog
from adaptivevision.persistence.database import open_database
from adaptivevision.persistence.integration import make_persistence_handler
from adaptivevision.persistence.repositories import SqliteResultRepository
from adaptivevision.preprocessing import (
    PreprocessingPipeline,
    PreprocessStep,
    ensure_grayscale,
    resize_to,
)
from adaptivevision.recipe import Recipe
from adaptivevision.recipe.store import JsonRecipeStore

#: Type of the ``on_result`` persistence hook.
OnResult = Callable[[InspectionResult], None]

#: Type of the preprocessing hook injected into the pipeline.
Preprocessor = Callable[[RawFrame], RawFrame]

#: Type of the rectification hook injected into the pipeline.
Rectifier = Callable[[RawFrame], RectifiedFrame]

#: Type of the alignment hook injected into the pipeline.
Aligner = Callable[[RectifiedFrame], LocalizedPart]


def build_camera(config: StationConfig) -> CameraDriver:
    """Build the camera driver for the station.

    Uses the null-object strategy: when no camera is configured, a synthetic
    :class:`NullCameraDriver` is returned so the walking skeleton runs without
    hardware.

    Args:
        config: The validated station configuration.

    Returns:
        A :class:`CameraDriver` ready to be opened.
    """
    if not config.cameras:
        # No camera configured: use a synthetic 640x480 null-object driver.
        from adaptivevision.common.enums import CameraKind
        from adaptivevision.config.settings import CameraConfig

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
    reference = load_golden_reference(str(reference_path))
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
    """Build an optional M9 anomaly detector from configuration.

    Wired only when ``MODEL_PATH`` names an ONNX model to load, following the
    same optional-hardware pattern as calibration/alignment: skeleton runs
    without a model configured remain possible. The threshold and the
    severity assigned to an anomaly (which the M10 policy turns into FAIL or
    REVIEW) come from the active recipe's declared decision policy when one
    is loaded, otherwise from safe defaults.

    Args:
        config: The validated station configuration.
        recipe: The active recipe, if any (see :func:`build_recipe`).

    Returns:
        A :class:`ThresholdAnomalyDetector` backed by ONNX Runtime, or
        ``None`` when no model is configured.
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
    """Build the M10 decision policy from the active recipe, if any.

    Args:
        recipe: The active recipe, if any (see :func:`build_recipe`).

    Returns:
        A policy translated from the recipe's declared decision contract via
        :meth:`DecisionPolicy.from_recipe`, or ``None`` when no recipe is
        configured -- the pipeline then falls back to its own
        any-defect-fails rule.
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
