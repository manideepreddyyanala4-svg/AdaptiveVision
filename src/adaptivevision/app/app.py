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

Nothing else in the codebase constructs these collaborators directly; they are
injected here and passed down.
"""

from __future__ import annotations

from collections.abc import Callable

from adaptivevision.acquisition.camera import NullCameraDriver
from adaptivevision.alignment import LocalizedPart, ReferenceAligner, load_golden_reference
from adaptivevision.app.station import StationController
from adaptivevision.calibration import CalibrationRectifier, load_calibration
from adaptivevision.common.interfaces import CameraDriver
from adaptivevision.common.result import InspectionResult
from adaptivevision.common.types import RawFrame, RectifiedFrame
from adaptivevision.config.settings import StationConfig
from adaptivevision.orchestration.pipeline import InspectionPipeline
from adaptivevision.orchestration.scheduler import InspectionScheduler
from adaptivevision.orchestration.state import StationStateMachine
from adaptivevision.orchestration.watchdog import CycleWatchdog
from adaptivevision.persistence.database import open_database
from adaptivevision.persistence.integration import make_persistence_handler
from adaptivevision.persistence.repositories import SqliteResultRepository
from adaptivevision.preprocessing import PreprocessingPipeline, ensure_grayscale

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
        A callable that preprocesses raw frames before calibration.
    """
    if config.extra.get("PREPROCESS_GRAYSCALE", True) is False:
        return PreprocessingPipeline().apply
    return PreprocessingPipeline((ensure_grayscale,)).apply


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


def build_station(config: StationConfig) -> StationController:
    """Assemble the full station from validated configuration.

    Args:
        config: The validated station configuration.

    Returns:
        A fully wired :class:`StationController`.
    """
    camera = build_camera(config)
    camera.open()

    pipeline = InspectionPipeline(
        camera,
        station_id=config.station_id,
        recipe_ver=config.default_recipe_id or "unset",
        preprocessor=build_preprocessor(config),
        rectifier=build_rectifier(config),
        aligner=build_aligner(config),
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
