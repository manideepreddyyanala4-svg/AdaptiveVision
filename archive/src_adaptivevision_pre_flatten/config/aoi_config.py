"""AOI settings: metrology calibration, drift thresholds, KPI targets
(Milestone M21).

:mod:`adaptivevision.config.loader` deliberately stays flat and env-var based
(``ADAPTIVEVISION_*`` -> :attr:`~adaptivevision.config.settings.StationConfig.extra`)
-- a good fit for a handful of scalar station settings, a poor one for a
nested tree of calibration/threshold values. Those live in
``configs/config.yaml`` instead and are loaded here, independently of
:class:`~adaptivevision.config.settings.StationConfig`, so both
``src/adaptivevision`` call sites and standalone scripts (e.g.
``scripts/evaluate_kpis.py``) can load them without going through the station
composition root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Repository-relative default location of the AOI settings file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


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
    if "threshold_percentile" in metrology_raw:
        percentile = metrology_raw["threshold_percentile"]
    else:
        percentile = default_metrology.threshold_percentile

    return AoiConfig(
        metrology=MetrologySettings(
            pixel_to_micron=float(
                metrology_raw.get("pixel_to_micron", default_metrology.pixel_to_micron)
            ),
            min_area_px2=int(
                metrology_raw.get("min_area_px2", default_metrology.min_area_px2)
            ),
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
