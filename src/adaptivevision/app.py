"""The composition root: wires every seam to a concrete implementation and assembles the station.

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

import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from adaptivevision.camera import (
    CalibrationRectifier,
    FileImageCameraDriver,
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
    DEFAULT_LOT_ID,
    AdaptiveVisionError,
    AnomalyDetector,
    CameraDriver,
    CameraKind,
    CommsError,
    ExecutionProvider,
    InspectionResult,
    RawFrame,
    RectifiedFrame,
    Severity,
    StationState,
)
from adaptivevision.communication import (
    DEFAULT_REJECT_COIL,
    ModbusTcpTransport,
    RejectDispatcher,
    SocketModbusClient,
)
from adaptivevision.config import (
    CameraConfig,
    JsonRecipeStore,
    Recipe,
    StationConfig,
    load_aoi_config,
)
from adaptivevision.decision import DecisionPolicy
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.explanation import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_STOP_TIMEOUT_S,
    AdvisoryWorker,
    OllamaAdvisoryEngine,
)
from adaptivevision.metrology import (
    MetrologyConfig,
    NormalPatchBank,
    ThresholdAnomalyDetector,
)
from adaptivevision.orchestration import (
    CycleWatchdog,
    InspectionPipeline,
    InspectionScheduler,
    ManualInspectionService,
    StationStateMachine,
)
from adaptivevision.storage import (
    LocalImageStore,
    SqliteAdvisoryRepository,
    SqliteResultRepository,
    make_persistence_handler,
    open_database,
)

logger = logging.getLogger(__name__)

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
        advisory_worker: Optional advisory worker, started on boot and
            stopped on shutdown so its thread never outlives the station.
    """

    def __init__(
        self,
        state_machine: StationStateMachine,
        pipeline: InspectionPipeline,
        scheduler: InspectionScheduler,
        watchdog: CycleWatchdog,
        on_result: OnResult | None = None,
        advisory_worker: AdvisoryWorker | None = None,
    ) -> None:
        """Initialize the controller with its collaborators."""
        self._state = state_machine
        self._pipeline = pipeline
        self._scheduler = scheduler
        self._watchdog = watchdog
        self._on_result = on_result
        self._advisory_worker = advisory_worker

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
        if self._advisory_worker is not None:
            self._advisory_worker.start()

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
        if self._advisory_worker is not None:
            self._advisory_worker.stop()


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
    walking skeleton runs without hardware. When ``DEMO_IMAGE_PATH`` is
    configured, a :class:`~adaptivevision.camera.FileImageCameraDriver`
    replaying that real image is returned instead -- for demonstrating a real
    trained model without physical camera hardware.

    Args:
        config: The validated station configuration.

    Returns:
        A :class:`~adaptivevision.common.CameraDriver` ready to be opened.
    """
    demo_image_path = config.extra.get("DEMO_IMAGE_PATH")
    if demo_image_path is not None:
        return FileImageCameraDriver(Path(str(demo_image_path)))

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


def build_persistence(
    config: StationConfig,
) -> tuple[SqliteResultRepository, OnResult, sessionmaker[Session]]:
    """Build the local persistence layer for the station.

    Args:
        config: The validated station configuration.

    Returns:
        A tuple of ``(repository, on_result_handler, session_factory)`` where
        the handler is a callable suitable for the station's ``on_result``
        hook and the factory backs any other repository sharing this database.
    """
    db_path = config.extra.get("DB_PATH", "adaptivevision.db")
    _, session_factory = open_database(db_path)
    repository = SqliteResultRepository(session_factory)
    handler = make_persistence_handler(repository)
    return repository, handler, session_factory


def build_image_store(config: StationConfig) -> LocalImageStore | None:
    """Build the optional archive of inspected frames.

    Args:
        config: The validated station configuration.

    Returns:
        A bounded :class:`~adaptivevision.storage.LocalImageStore` when
        ``IMAGE_STORE_DIR`` is configured, otherwise ``None`` -- a station
        that keeps no image trail still runs end-to-end.
    """
    directory = config.extra.get("IMAGE_STORE_DIR")
    if directory is None:
        return None
    max_images = int(config.extra.get("IMAGE_STORE_MAX", 1000))
    return LocalImageStore(str(directory), max_images=max_images)


def build_reject_dispatcher(config: StationConfig) -> RejectDispatcher | None:
    """Build the optional PLC reject-coil dispatcher.

    Args:
        config: The validated station configuration.

    Returns:
        A connected :class:`~adaptivevision.communication.RejectDispatcher`
        when ``PLC_HOST`` is configured, otherwise ``None``. A PLC that
        cannot be reached at boot is logged and skipped rather than fatal:
        the station must still inspect and record parts.
    """
    host = config.extra.get("PLC_HOST")
    if host is None:
        return None
    port = int(config.extra.get("PLC_PORT", 502))
    coil = int(config.extra.get("PLC_REJECT_COIL", DEFAULT_REJECT_COIL))
    transport = ModbusTcpTransport(SocketModbusClient(str(host), port))
    try:
        transport.connect()
    except CommsError as exc:
        logger.error(
            "PLC unavailable; reject dispatch disabled",
            extra={"host": host, "port": port, "error": str(exc)},
        )
        return None
    return RejectDispatcher(transport, coil_address=coil)


def build_advisory_worker(
    config: StationConfig,
    session_factory: sessionmaker[Session],
) -> AdvisoryWorker | None:
    """Build the optional off-thread advisory worker.

    Args:
        config: The validated station configuration.
        session_factory: Factory backing the advisory repository. Must be
            file-backed: the worker persists from its own thread.

    Returns:
        An :class:`~adaptivevision.explanation.AdvisoryWorker` when
        ``ADVISORY_ENABLED`` is set, otherwise ``None``.
    """
    enabled = str(config.extra.get("ADVISORY_ENABLED", "")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    engine = OllamaAdvisoryEngine(
        model=str(config.extra.get("OLLAMA_MODEL", DEFAULT_MODEL)),
        host=str(config.extra.get("OLLAMA_HOST", DEFAULT_HOST)),
        # The worker is off the critical path, so it can afford a budget the
        # synchronous default deliberately does not.
        timeout_s=float(config.extra.get("OLLAMA_TIMEOUT_S", 60.0)),
    )
    return AdvisoryWorker(
        engine,
        SqliteAdvisoryRepository(session_factory),
        category=str(config.extra.get("PRODUCT_CATEGORY", "")),
        stop_timeout=float(config.extra.get("ADVISORY_STOP_TIMEOUT_S", DEFAULT_STOP_TIMEOUT_S)),
    )


def chain_handlers(*handlers: OnResult | None) -> OnResult:
    """Compose several ``on_result`` consumers into one.

    Each handler runs in order; ``None`` entries are skipped. Handlers are
    expected to contain their own failures (persistence and reject dispatch
    both do), so this deliberately does not add another try/except layer.

    Args:
        handlers: The consumers to chain, in execution order.

    Returns:
        A single callable suitable for the station's ``on_result`` hook.
    """
    active = [h for h in handlers if h is not None]

    def dispatch(result: InspectionResult) -> None:
        for handler in active:
            handler(result)

    return dispatch


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
    grayscale = config.extra.get("PREPROCESS_GRAYSCALE", True)
    # extra's values come straight from environment strings (see
    # config.load_config), never real booleans, so "False"/"false"/"0" must
    # be recognized -- `is not False` would silently never match a string.
    if str(grayscale).strip().lower() not in {"false", "0", "no"}:
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
        metrology_config=build_metrology_config(config),
        patch_bank=build_patch_bank(config),
    )


def build_patch_bank(config: StationConfig) -> NormalPatchBank | None:
    """Load the normal-patch reference bank used to localize defects.

    Args:
        config: The validated station configuration.

    Returns:
        The bank named by ``PATCH_BANK_PATH``, or ``None`` when unset or
        unreadable. Localization is optional; a station without a bank still
        produces verdicts, just no defect geometry (see
        ``scripts/build_patch_bank.py`` to create one).
    """
    bank_path = config.extra.get("PATCH_BANK_PATH")
    if bank_path is None:
        return None
    try:
        return NormalPatchBank.load(str(bank_path))
    except (OSError, ValueError) as exc:
        logger.error(
            "Patch bank unavailable; defect localization disabled",
            extra={"path": str(bank_path), "error": str(exc)},
        )
        return None


def build_metrology_config(config: StationConfig) -> MetrologyConfig | None:
    """Build the defect-metrology configuration, if enabled.

    Calibration lives in the AOI settings file (``configs/config.yaml``),
    not the environment: ``pixel_to_micron`` is a physical property of the
    optics, versioned with the rest of the inspection setup.

    Args:
        config: The validated station configuration.

    Returns:
        A :class:`~adaptivevision.metrology.MetrologyConfig` when
        ``METROLOGY_ENABLED`` is set, otherwise ``None`` -- a station that
        only needs a verdict skips localization entirely.
    """
    enabled = str(config.extra.get("METROLOGY_ENABLED", "")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    settings = load_aoi_config().metrology
    return MetrologyConfig(
        pixel_to_micron=settings.pixel_to_micron,
        min_area_px2=settings.min_area_px2,
        threshold_percentile=settings.threshold_percentile,
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


def build_manual_inspector(
    config: StationConfig,
) -> tuple[ManualInspectionService, SqliteResultRepository, sessionmaker[Session]]:
    """Assemble the on-demand (teach-mode) inspection service.

    Builds the same collaborators :func:`build_station` does -- model,
    patch bank, metrology, recipe, decision policy, image archive,
    persistence, PLC dispatch, advisory worker -- so a manually submitted
    sample takes an identical path to a triggered part. The costly ones
    (inference engine, patch bank) are built once and shared; only the
    per-cycle pipeline is rebuilt per request.

    Args:
        config: The validated station configuration.

    Returns:
        ``(service, repository, session_factory)``. The repository and
        factory are returned so the caller can serve the same database
        through the API without opening a second one.
    """
    recipe = build_recipe(config)
    image_store = build_image_store(config)
    preprocessor = build_preprocessor(config)
    rectifier = build_rectifier(config)
    aligner = build_aligner(config)
    anomaly_detector = build_anomaly_detector(config, recipe)
    decision_policy = build_decision_policy(recipe)
    lot_id = str(config.extra.get("LOT_ID", DEFAULT_LOT_ID))

    repository, persist, session_factory = build_persistence(config)
    reject = build_reject_dispatcher(config)
    advisory_worker = build_advisory_worker(config, session_factory)
    if advisory_worker is not None:
        advisory_worker.start()

    on_result = chain_handlers(
        persist,
        reject.on_result if reject is not None else None,
        advisory_worker.on_result if advisory_worker is not None else None,
    )

    def pipeline_factory(camera: CameraDriver) -> InspectionPipeline:
        return InspectionPipeline(
            camera,
            station_id=config.station_id,
            recipe_ver=config.default_recipe_id or "unset",
            preprocessor=preprocessor,
            rectifier=rectifier,
            aligner=aligner,
            recipe=recipe,
            anomaly_detector=anomaly_detector,
            decision_policy=decision_policy,
            image_store=image_store,
            lot_id=lot_id,
        )

    return ManualInspectionService(pipeline_factory, on_result), repository, session_factory


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
    image_store = build_image_store(config)

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
        image_store=image_store,
        lot_id=str(config.extra.get("LOT_ID", DEFAULT_LOT_ID)),
    )
    scheduler = InspectionScheduler(pipeline)
    watchdog = CycleWatchdog(config.cycle_timeout_ms)
    state_machine = StationStateMachine()

    _, persist, session_factory = build_persistence(config)
    reject = build_reject_dispatcher(config)
    advisory_worker = build_advisory_worker(config, session_factory)

    # Order matters: the result is durable before anything physical or
    # explanatory happens to it.
    on_result = chain_handlers(
        persist,
        reject.on_result if reject is not None else None,
        advisory_worker.on_result if advisory_worker is not None else None,
    )

    return StationController(
        state_machine=state_machine,
        pipeline=pipeline,
        scheduler=scheduler,
        watchdog=watchdog,
        on_result=on_result,
        advisory_worker=advisory_worker,
    )
