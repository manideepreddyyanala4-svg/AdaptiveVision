"""Unit tests for :mod:`adaptivevision.config` (Milestone M21)."""

from __future__ import annotations

from pathlib import Path

from adaptivevision.config import AoiConfig, DriftSettings, KpiSettings, MetrologySettings
from adaptivevision.config import load_aoi_config


def test_load_aoi_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_aoi_config(tmp_path / "does-not-exist.yaml")
    assert config == AoiConfig()
    assert config.metrology == MetrologySettings()
    assert config.drift == DriftSettings()
    assert config.kpi == KpiSettings()


def test_load_aoi_config_empty_file_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_aoi_config(path) == AoiConfig()


def test_load_aoi_config_reads_all_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
metrology:
  pixel_to_micron: 2.5
  min_area_px2: 9
  threshold_percentile: 95.0
drift:
  window_size: 50
  p_value_threshold: 0.05
kpi:
  target_escape_rate: 0.002
""",
        encoding="utf-8",
    )

    config = load_aoi_config(path)

    assert config.metrology == MetrologySettings(
        pixel_to_micron=2.5, min_area_px2=9, threshold_percentile=95.0
    )
    assert config.drift == DriftSettings(window_size=50, p_value_threshold=0.05)
    assert config.kpi == KpiSettings(target_escape_rate=0.002)


def test_load_aoi_config_partial_file_fills_in_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("metrology:\n  pixel_to_micron: 3.0\n", encoding="utf-8")

    config = load_aoi_config(path)

    assert config.metrology.pixel_to_micron == 3.0
    assert config.metrology.min_area_px2 == MetrologySettings().min_area_px2
    # Regression check: an absent threshold_percentile key must fall back to
    # the field's own default, not get silently hard-coded to None (a real
    # bug this test caught when that default changed from None to 99.5).
    assert config.metrology.threshold_percentile == MetrologySettings().threshold_percentile
    assert config.drift == DriftSettings()
    assert config.kpi == KpiSettings()


def test_load_aoi_config_null_threshold_percentile_stays_none(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("metrology:\n  threshold_percentile: null\n", encoding="utf-8")

    config = load_aoi_config(path)

    assert config.metrology.threshold_percentile is None


def test_load_aoi_config_default_path_reads_real_repo_config() -> None:
    # configs/config.yaml ships in the repo; this is the one test that
    # exercises the real, no-argument default path end to end.
    config = load_aoi_config()
    assert config.metrology.pixel_to_micron > 0
    assert 0.0 < config.kpi.target_escape_rate < 1.0
