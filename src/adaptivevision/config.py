"""Settings: station configuration, AOI calibration/drift/KPI settings, and product recipes.

Configuration is a cross-cutting concern read once at startup and injected
into whatever needs it. Three related sources, each with a different shape
because they change on different timescales and by different people: station
settings (:class:`StationConfig`) come from environment variables read by the
composition root at every boot; AOI settings (:class:`AoiConfig`) come from
``configs/config.yaml``, a nested settings tree env vars don't fit well; and
a :class:`Recipe` is the versioned, per-product-variant specification an
operator loads and swaps between inspection runs.

Unknown ``ADAPTIVEVISION_*`` environment variables are preserved in
:attr:`StationConfig.extra` rather than rejected, so new settings can be
added without breaking existing deployments.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml

from adaptivevision.common import (
    ROI,
    CameraKind,
    ExecutionProvider,
    MeasurementSpec,
    RecipeError,
    RecipeStore,
    Severity,
)

# =============================================================================
# Station configuration
#
# Owns the *shape* of the station's runtime configuration and the validation
# rules applied to it. The composition root (app.py) reads raw values from
# the environment and builds a validated StationConfig here.
# =============================================================================


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Configuration for a single camera device.

    Attributes:
        camera_id: Stable identifier of the camera.
        kind: Sensor modality.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        fps: Requested capture rate in frames per second.
        device: Backend-specific device reference (for example a serial number
            or index), if any.
    """

    camera_id: str
    kind: CameraKind
    width: int
    height: int
    fps: float
    device: str | None = None

    def __post_init__(self) -> None:
        """Validate numeric invariants."""
        if self.width <= 0:
            msg = "CameraConfig.width must be positive"
            raise ValueError(msg)
        if self.height <= 0:
            msg = "CameraConfig.height must be positive"
            raise ValueError(msg)
        if self.fps <= 0.0:
            msg = "CameraConfig.fps must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StationConfig:
    """Validated, immutable station configuration.

    Attributes:
        station_id: Stable identifier of this station.
        log_level: Root logging level name (for example ``"INFO"``).
        cameras: Configured camera devices, keyed by ``camera_id``.
        default_recipe_id: Recipe to load at startup, if any.
        execution_provider: Preferred ONNX Runtime execution provider.
        cycle_timeout_ms: Maximum allowed inspection cycle time in milliseconds.
        extra: Unrecognized settings preserved for forward compatibility.
    """

    station_id: str
    log_level: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    default_recipe_id: str | None = None
    execution_provider: ExecutionProvider = ExecutionProvider.CPU
    cycle_timeout_ms: float = 5000.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants."""
        if not self.station_id:
            msg = "StationConfig.station_id must not be empty"
            raise ValueError(msg)
        if self.cycle_timeout_ms <= 0.0:
            msg = "StationConfig.cycle_timeout_ms must be positive"
            raise ValueError(msg)

    def camera(self, camera_id: str) -> CameraConfig:
        """Return the camera with ``camera_id``.

        Args:
            camera_id: Identifier of the camera to look up.

        Returns:
            The matching :class:`CameraConfig`.

        Raises:
            KeyError: If no camera with ``camera_id`` is configured.
        """
        return self.cameras[camera_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "station_id": self.station_id,
            "log_level": self.log_level,
            "cameras": {cid: _camera_to_dict(c) for cid, c in self.cameras.items()},
            "default_recipe_id": self.default_recipe_id,
            "execution_provider": self.execution_provider.value,
            "cycle_timeout_ms": self.cycle_timeout_ms,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        cameras = {
            cid: CameraConfig(
                camera_id=c["camera_id"],
                kind=CameraKind(c["kind"]),
                width=c["width"],
                height=c["height"],
                fps=c["fps"],
                device=c.get("device"),
            )
            for cid, c in data.get("cameras", {}).items()
        }
        return cls(
            station_id=data["station_id"],
            log_level=data["log_level"],
            cameras=cameras,
            default_recipe_id=data.get("default_recipe_id"),
            execution_provider=ExecutionProvider(data.get("execution_provider", "cpu")),
            cycle_timeout_ms=data.get("cycle_timeout_ms", 5000.0),
            extra=dict(data.get("extra", {})),
        )


def _camera_to_dict(camera: CameraConfig) -> dict[str, Any]:
    """Serialize a single :class:`CameraConfig`."""
    return {
        "camera_id": camera.camera_id,
        "kind": camera.kind.value,
        "width": camera.width,
        "height": camera.height,
        "fps": camera.fps,
        "device": camera.device,
    }


#: Prefix for all AdaptiveVision environment variables.
_ENV_PREFIX = "ADAPTIVEVISION_"

#: Default station identifier when none is configured.
_DEFAULT_STATION_ID = "station-01"


def load_env_file(path: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` ``.env`` file into a dictionary.

    Existing environment variables take precedence: a key already present in
    ``environ`` is not overridden by the file. Blank lines and lines starting
    with ``#`` are ignored. Values may be quoted with single or double quotes.

    Args:
        path: Path to the ``.env`` file.
        environ: Environment mapping to consult for precedence. Defaults to
            :data:`os.environ`.

    Returns:
        A dictionary of parsed key/value pairs (only those not already set).
    """
    env = os.environ if environ is None else environ
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in env:
            parsed[key] = value
    return parsed


def load_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> StationConfig:
    """Load and validate station configuration.

    Args:
        environ: Environment mapping to read from. Defaults to
            :data:`os.environ`.
        env_file: Optional ``.env`` file to merge in (existing environment
            variables win). Defaults to ``.env`` in the current directory.

    Returns:
        A validated :class:`StationConfig`.

    Raises:
        ValueError: If a configured value is invalid.
    """
    env = os.environ if environ is None else environ
    merged: dict[str, str] = dict(env)
    resolved_env_file = Path(".env") if env_file is None else env_file
    merged.update(load_env_file(resolved_env_file, environ=env))

    station_id = merged.get(f"{_ENV_PREFIX}STATION_ID", _DEFAULT_STATION_ID)
    log_level = merged.get(f"{_ENV_PREFIX}LOG_LEVEL", "INFO")
    default_recipe_id = merged.get(f"{_ENV_PREFIX}DEFAULT_RECIPE_ID")
    provider_name = merged.get(f"{_ENV_PREFIX}EXECUTION_PROVIDER", "cpu")
    cycle_timeout_ms = _parse_float(
        merged.get(f"{_ENV_PREFIX}CYCLE_TIMEOUT_MS", "5000.0"),
        f"{_ENV_PREFIX}CYCLE_TIMEOUT_MS",
    )

    cameras = _parse_cameras(merged)

    extra = {
        key[len(_ENV_PREFIX) :]: value
        for key, value in merged.items()
        if key.startswith(_ENV_PREFIX)
        and key
        not in {
            f"{_ENV_PREFIX}STATION_ID",
            f"{_ENV_PREFIX}LOG_LEVEL",
            f"{_ENV_PREFIX}DEFAULT_RECIPE_ID",
            f"{_ENV_PREFIX}EXECUTION_PROVIDER",
            f"{_ENV_PREFIX}CYCLE_TIMEOUT_MS",
            f"{_ENV_PREFIX}CAMERA_IDS",
        }
    }

    return StationConfig(
        station_id=station_id,
        log_level=log_level,
        cameras=cameras,
        default_recipe_id=default_recipe_id,
        execution_provider=ExecutionProvider(provider_name),
        cycle_timeout_ms=cycle_timeout_ms,
        extra=extra,
    )


def _parse_cameras(merged: Mapping[str, str]) -> dict[str, CameraConfig]:
    """Parse camera configurations from ``ADAPTIVEVISION_CAMERA_IDS``.

    Each camera id in the comma-separated ``ADAPTIVEVISION_CAMERA_IDS`` list is
    expanded with ``ADAPTIVEVISION_CAMERA_<ID>_*`` variables.

    Args:
        merged: Merged environment mapping.

    Returns:
        A dictionary of :class:`CameraConfig` keyed by camera id.
    """
    raw_ids = merged.get(f"{_ENV_PREFIX}CAMERA_IDS", "")
    cameras: dict[str, CameraConfig] = {}
    for camera_id in (part.strip() for part in raw_ids.split(",") if part.strip()):
        prefix = f"{_ENV_PREFIX}CAMERA_{camera_id.upper()}_"
        kind_name = merged.get(f"{prefix}KIND", "area_scan_2d")
        width = _parse_int(merged.get(f"{prefix}WIDTH", "640"), f"{prefix}WIDTH")
        height = _parse_int(merged.get(f"{prefix}HEIGHT", "480"), f"{prefix}HEIGHT")
        fps = _parse_float(merged.get(f"{prefix}FPS", "30.0"), f"{prefix}FPS")
        device = merged.get(f"{prefix}DEVICE")
        cameras[camera_id] = CameraConfig(
            camera_id=camera_id,
            kind=CameraKind(kind_name),
            width=width,
            height=height,
            fps=fps,
            device=device,
        )
    return cameras


def _parse_int(raw: str, name: str) -> int:
    """Parse ``raw`` as an integer, raising a descriptive error on failure."""
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"Invalid integer for {name}: {raw!r}"
        raise ValueError(msg) from exc


def _parse_float(raw: str, name: str) -> float:
    """Parse ``raw`` as a float, raising a descriptive error on failure."""
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"Invalid number for {name}: {raw!r}"
        raise ValueError(msg) from exc


# =============================================================================
# AOI settings: metrology calibration, drift thresholds, KPI targets
# (Milestone M21)
#
# StationConfig above deliberately stays flat and env-var based -- a good fit
# for a handful of scalar station settings, a poor one for a nested tree of
# calibration/threshold values. Those live in configs/config.yaml instead and
# are loaded here, independently of StationConfig, so both src/adaptivevision
# call sites and standalone scripts (e.g. scripts/evaluate_kpis.py) can load
# them without going through the station composition root.
# =============================================================================

#: Repository-relative default location of the AOI settings file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass(frozen=True, slots=True)
class MetrologySettings:
    """Defect-metrology calibration and thresholding settings.

    Attributes:
        pixel_to_micron: Physical size, in microns, of one heatmap pixel's
            edge. Must be positive.
        min_area_px2: Regions smaller than this are discarded as noise.
        threshold_percentile: When set, threshold at this percentile of the
            heatmap's own values instead of computing an Otsu threshold.

    The ``min_area_px2=100``/``threshold_percentile=99.5`` defaults are
    empirically tuned (see ``configs/config.yaml``'s comments) against the
    dashboard's self-similarity heatmap on MVTec bottle at 256x256, not a
    theoretical ideal -- Otsu's bimodal assumption doesn't fit that signal
    well, and a percentile alone can never report zero regions for any
    image, clean or not (it always keeps its own top slice by definition).
    A different heatmap source or resolution may need different values.
    """

    pixel_to_micron: float = 1.0
    min_area_px2: int = 100
    threshold_percentile: float | None = 99.5


@dataclass(frozen=True, slots=True)
class DriftSettings:
    """Sensor/illumination drift-detector settings.

    Attributes:
        window_size: Sliding-window size, in most-recent inspections.
        p_value_threshold: KS-test p-value below which drift is flagged.
    """

    window_size: int = 100
    p_value_threshold: float = 0.01


@dataclass(frozen=True, slots=True)
class KpiSettings:
    """Industrial yield KPI targets.

    Attributes:
        target_escape_rate: Maximum acceptable fraction of defective parts
            misclassified as PASS when searching for an operating threshold.
    """

    target_escape_rate: float = 0.001


@dataclass(frozen=True, slots=True)
class AoiConfig:
    """Root AOI settings object, as loaded from ``configs/config.yaml``.

    Attributes:
        metrology: Defect-metrology settings.
        drift: Drift-detector settings.
        kpi: Yield KPI targets.
    """

    metrology: MetrologySettings = field(default_factory=MetrologySettings)
    drift: DriftSettings = field(default_factory=DriftSettings)
    kpi: KpiSettings = field(default_factory=KpiSettings)


def load_aoi_config(path: str | Path | None = None) -> AoiConfig:
    """Load AOI settings from a YAML file, falling back to defaults.

    Args:
        path: Path to the settings YAML file. Defaults to
            :data:`DEFAULT_CONFIG_PATH` (``configs/config.yaml`` at the repo
            root).

    Returns:
        The populated :class:`AoiConfig`. Missing keys and a missing file
        both fall back to each setting's documented default -- there is no
        required configuration, only an optional override.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return AoiConfig()

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    metrology_raw: dict[str, Any] = raw.get("metrology") or {}
    drift_raw: dict[str, Any] = raw.get("drift") or {}
    kpi_raw: dict[str, Any] = raw.get("kpi") or {}

    default_metrology = MetrologySettings()
    default_drift = DriftSettings()
    default_kpi = KpiSettings()

    # An absent key must fall back to the dataclass default (not hard-coded
    # None): dict.get() can't tell "key absent" from "key explicitly null"
    # apart since both read back as None, but they mean different things --
    # explicit null means "use Otsu, not a percentile," while an absent key
    # means "use whatever this field's own default is."
    if "threshold_percentile" in metrology_raw:  # noqa: SIM401 -- explicit on purpose, see comment above
        percentile = metrology_raw["threshold_percentile"]
    else:
        percentile = default_metrology.threshold_percentile

    return AoiConfig(
        metrology=MetrologySettings(
            pixel_to_micron=float(
                metrology_raw.get("pixel_to_micron", default_metrology.pixel_to_micron)
            ),
            min_area_px2=int(metrology_raw.get("min_area_px2", default_metrology.min_area_px2)),
            threshold_percentile=float(percentile) if percentile is not None else None,
        ),
        drift=DriftSettings(
            window_size=int(drift_raw.get("window_size", default_drift.window_size)),
            p_value_threshold=float(
                drift_raw.get("p_value_threshold", default_drift.p_value_threshold)
            ),
        ),
        kpi=KpiSettings(
            target_escape_rate=float(
                kpi_raw.get("target_escape_rate", default_kpi.target_escape_rate)
            ),
        ),
    )


#: File extension used for stored recipes.
_RECIPE_SUFFIX = ".json"


# =============================================================================
# Recipe: the per-product-variant inspection specification
#
# A Recipe is the immutable specification for inspecting one product variant.
# It composes the shared domain vocabulary (ROI regions, MeasurementSpec /
# Tolerance bands, the decision enums) plus a set of inspector references.
# Inspector references are validated strings resolved against a registry
# rather than a new enum, so extending the inspector set needs no spec
# change. Invalid recipes raise RecipeError, which is non-recoverable and
# drives the station to a fault / safe state.
# =============================================================================

#: A registry of known inspector names, keyed by the string a recipe uses.
InspectorRegistry = frozenset[str]


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Rules that map inspection outcomes to a final verdict.

    This is a *declared* policy carried by the recipe; the logic that
    applies it lives in ``decision.py``'s ``DecisionPolicy`` (same name,
    different layer: this one is data, that one is behavior). The fields are
    the stable contract the decision engine consumes.

    Attributes:
        anomaly_threshold: Score above which a part is flagged anomalous.
        review_on_anomaly: Whether an anomaly forces ``REVIEW`` rather than
            ``FAIL``.
        max_defects: Maximum tolerated defect count before ``FAIL``.
        fail_severity: Minimum severity that forces ``FAIL``.
    """

    anomaly_threshold: float = 0.5
    review_on_anomaly: bool = False
    max_defects: int = 0
    fail_severity: Severity = Severity.MAJOR

    def __post_init__(self) -> None:
        """Validate policy invariants."""
        if not 0.0 <= self.anomaly_threshold <= 1.0:
            msg = "DecisionPolicy.anomaly_threshold must be in [0, 1]"
            raise RecipeError(msg)
        if self.max_defects < 0:
            msg = "DecisionPolicy.max_defects must be non-negative"
            raise RecipeError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "anomaly_threshold": self.anomaly_threshold,
            "review_on_anomaly": self.review_on_anomaly,
            "max_defects": self.max_defects,
            "fail_severity": self.fail_severity.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            anomaly_threshold=data.get("anomaly_threshold", 0.5),
            review_on_anomaly=data.get("review_on_anomaly", False),
            max_defects=data.get("max_defects", 0),
            fail_severity=Severity(data.get("fail_severity", "major")),
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    """The immutable specification for inspecting one product variant.

    Attributes:
        recipe_id: Stable identifier of the recipe.
        version: Version string of this recipe revision.
        rois: Regions of interest inspected by this recipe.
        measurement_specs: Dimensional specifications to evaluate.
        inspectors: Validated inspector names to run, in order.
        decision: Decision policy for this recipe.
        product_name: Optional human-readable product name.
    """

    recipe_id: str
    version: str
    rois: tuple[ROI, ...] = ()
    measurement_specs: tuple[MeasurementSpec, ...] = ()
    inspectors: tuple[str, ...] = ()
    decision: DecisionPolicy = field(default_factory=DecisionPolicy)
    product_name: str | None = None

    def __post_init__(self) -> None:
        """Validate recipe invariants."""
        if not self.recipe_id:
            msg = "Recipe.recipe_id must not be empty"
            raise RecipeError(msg)
        if not self.version:
            msg = "Recipe.version must not be empty"
            raise RecipeError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "recipe_id": self.recipe_id,
            "version": self.version,
            "rois": [roi.to_dict() for roi in self.rois],
            "measurement_specs": [spec.to_dict() for spec in self.measurement_specs],
            "inspectors": list(self.inspectors),
            "decision": self.decision.to_dict(),
            "product_name": self.product_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            recipe_id=data["recipe_id"],
            version=data["version"],
            rois=tuple(ROI.from_dict(r) for r in data.get("rois", ())),
            measurement_specs=tuple(
                MeasurementSpec.from_dict(s) for s in data.get("measurement_specs", ())
            ),
            inspectors=tuple(data.get("inspectors", ())),
            decision=DecisionPolicy.from_dict(data.get("decision", {})),
            product_name=data.get("product_name"),
        )


def validate_inspectors(
    inspectors: tuple[str, ...],
    registry: InspectorRegistry,
) -> tuple[str, ...]:
    """Validate inspector names against ``registry``.

    Args:
        inspectors: Inspector names to validate.
        registry: Set of known inspector names.

    Returns:
        The validated inspector names, deduplicated while preserving order.

    Raises:
        RecipeError: If any inspector name is not present in ``registry``.
    """
    seen: set[str] = set()
    validated: list[str] = []
    for name in inspectors:
        if name not in registry:
            msg = f"Unknown inspector: {name!r}"
            raise RecipeError(msg)
        if name not in seen:
            seen.add(name)
            validated.append(name)
    return tuple(validated)


class JsonRecipeStore(RecipeStore[Recipe]):
    """A :class:`~adaptivevision.common.RecipeStore` backed by JSON files on disk.

    Each recipe is stored as ``<recipe_id>.json`` in the configured directory.
    The store is not thread-safe; the orchestration layer serializes access.

    Args:
        directory: Directory in which recipe files are stored. Created on
            first write if it does not exist.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the store with a backing directory."""
        self._directory = directory

    def load(self, recipe_id: str) -> Recipe:
        """Load a recipe by identifier.

        Args:
            recipe_id: Identifier of the recipe to load.

        Returns:
            The loaded :class:`Recipe`.

        Raises:
            RecipeError: If the recipe is missing or invalid.
        """
        path = self._path_for(recipe_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            msg = f"Recipe not found: {recipe_id!r}"
            raise RecipeError(msg) from exc
        except (json.JSONDecodeError, OSError) as exc:
            msg = f"Failed to read recipe {recipe_id!r}: {exc}"
            raise RecipeError(msg) from exc
        return self._from_dict(data, recipe_id)

    def save(self, recipe: Recipe) -> None:
        """Persist a recipe.

        Args:
            recipe: The recipe to persist.

        Raises:
            RecipeError: On storage failure.
        """
        path = self._path_for(recipe.recipe_id)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(recipe.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            msg = f"Failed to write recipe {recipe.recipe_id!r}: {exc}"
            raise RecipeError(msg) from exc

    def list_ids(self) -> tuple[str, ...]:
        """Return the identifiers of all stored recipes.

        Returns:
            A tuple of recipe identifiers, sorted lexically.
        """
        if not self._directory.exists():
            return ()
        ids: list[str] = []
        for path in self._directory.glob(f"*{_RECIPE_SUFFIX}"):
            ids.append(path.stem)
        return tuple(sorted(ids))

    def _path_for(self, recipe_id: str) -> Path:
        """Return the file path for ``recipe_id``."""
        return self._directory / f"{recipe_id}{_RECIPE_SUFFIX}"

    @staticmethod
    def _from_dict(data: dict[str, Any], recipe_id: str) -> Recipe:
        """Deserialize a recipe, verifying the id matches the file name."""
        try:
            recipe = Recipe.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"Invalid recipe {recipe_id!r}: {exc}"
            raise RecipeError(msg) from exc
        if recipe.recipe_id != recipe_id:
            msg = f"Recipe id mismatch: file {recipe_id!r}, content {recipe.recipe_id!r}"
            raise RecipeError(msg)
        return recipe
