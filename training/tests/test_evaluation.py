"""Correctness checks for the new metrics (AUPIMO, PG2, PB2) against known values."""

from __future__ import annotations

import numpy as np
import pytest
from training.evaluate import compute_aupimo, compute_metrics

# ---------------------------------------------------------------------------
# PG2 / PB2
# ---------------------------------------------------------------------------


def test_pg2_pb2_perfect_separation_is_one() -> None:
    """Normal and anomalous scores fully separated -> both metrics hit 1.0."""
    normal_scores = np.arange(100, dtype=float)  # 0..99
    anomalous_scores = np.arange(100, 200, dtype=float)  # 100..199
    scores = np.concatenate([normal_scores, anomalous_scores])
    labels = np.concatenate([np.zeros(100, dtype=bool), np.ones(100, dtype=bool)])

    metrics = compute_metrics(scores, labels)
    assert metrics.pg2 == pytest.approx(1.0)
    assert metrics.pb2 == pytest.approx(1.0)


def test_pg2_pb2_identical_distributions_match_the_presort_rate() -> None:
    """Normal and anomalous scores drawn from the identical value set: with
    numpy's linear percentile interpolation, the threshold at the 2nd
    percentile of one class admits almost exactly 2% of the other -- a
    hand-derivable known value, not just a bound.
    """
    values = np.arange(100, dtype=float)  # 0..99, used for both classes
    scores = np.concatenate([values, values])
    labels = np.concatenate([np.zeros(100, dtype=bool), np.ones(100, dtype=bool)])

    metrics = compute_metrics(scores, labels)
    # percentile(arange(100), 2) == 1.98 -> normal_scores < 1.98 are {0, 1} -> 2/100
    assert metrics.pg2 == pytest.approx(0.02, abs=1e-9)
    # percentile(arange(100), 98) == 97.02 -> anomalous_scores >= 97.02 are {98, 99} -> 2/100
    assert metrics.pb2 == pytest.approx(0.02, abs=1e-9)


def test_pg2_pb2_bounded_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = np.concatenate([np.zeros(100, dtype=bool), np.ones(100, dtype=bool)])
    metrics = compute_metrics(scores, labels)
    assert 0.0 <= metrics.pg2 <= 1.0
    assert 0.0 <= metrics.pb2 <= 1.0


def test_pg2_pb2_nan_when_only_one_class_present() -> None:
    scores = np.arange(10, dtype=float)
    labels = np.zeros(10, dtype=bool)  # every image normal
    metrics = compute_metrics(scores, labels)
    assert metrics.pg2 != metrics.pg2  # NaN
    assert metrics.pb2 != metrics.pb2


# ---------------------------------------------------------------------------
# AUPIMO
# ---------------------------------------------------------------------------


def _step_function_case(size: int = 20, region: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """One image, a clean step-function map: 1.0 inside a region, 0.0 outside.

    Deliberately degenerate (only two distinct map values) -- exercised by
    test_aupimo_exact_step_function_integrates_to_zero below to document the
    edge case, not used for the "near-perfect detector" check (see
    test_aupimo_near_perfect_detector_close_to_one), since an exact step
    function gives a zero-width FPR range that trapezoidal integration
    correctly-but-unhelpfully integrates to 0 -- inherited from the existing
    compute_aupro's integration method, not something specific to AUPIMO.
    """
    mask = np.zeros((1, size, size), dtype=bool)
    mask[0, :region, :region] = True
    maps = mask.astype(np.float32)
    return maps, mask


def test_aupimo_exact_step_function_integrates_to_zero() -> None:
    """Documents the degenerate case above rather than treating it as a bug:
    a map with only two distinct values gives a zero-width FPR range, and
    trapezoidal integration over zero width is 0 regardless of the (perfect)
    overlap -- the same behavior compute_aupro already has, ported as-is."""
    maps, mask = _step_function_case()
    mean_score, _ = compute_aupimo(maps, mask)
    assert mean_score == pytest.approx(0.0, abs=1e-9)


def test_aupimo_near_perfect_detector_close_to_one() -> None:
    """A realistic near-perfect map -- continuous values spread widely enough
    for the threshold sweep to resolve the FPR transition (unlike the narrow
    band above) -- should score close to the 1.0 upper bound.
    """
    rng = np.random.default_rng(0)
    size = 40
    mask = np.zeros((1, size, size), dtype=bool)
    mask[0, :12, :12] = True
    maps = rng.uniform(0.0, 0.5, size=(1, size, size)).astype(np.float32)
    maps[0, :12, :12] = rng.uniform(0.5, 1.0, size=(12, 12)).astype(np.float32)

    mean_score, per_image = compute_aupimo(maps, mask)
    assert mean_score > 0.95
    assert len(per_image) == 1


def test_aupimo_uncorrelated_noise_is_low() -> None:
    rng = np.random.default_rng(0)
    size = 32
    mask = np.zeros((5, size, size), dtype=bool)
    for i in range(5):
        mask[i, 4:10, 4:10] = True  # a fixed defect region in every image
    maps = rng.uniform(size=(5, size, size)).astype(np.float32)  # noise, uncorrelated with mask

    mean_score, per_image = compute_aupimo(maps, mask)
    # A detector no better than chance should sit near the FPR-limit-normalized
    # baseline, i.e. well below a real detector's score -- not asserting an
    # exact value (this is randomized), just that it's far from perfect.
    assert mean_score < 0.9
    assert len(per_image) == 5


def test_aupimo_skips_images_with_no_defect() -> None:
    """An all-normal image contributes nothing -- mirrors AUPRO's own-image
    nan guard for a set with no anomalous pixels at all."""
    size = 16
    maps = np.random.default_rng(0).uniform(size=(3, size, size)).astype(np.float32)
    mask = np.zeros((3, size, size), dtype=bool)
    mask[0, 2:6, 2:6] = True
    # images 1, 2 have no defect at all

    _, per_image = compute_aupimo(maps, mask)
    assert len(per_image) == 1  # only image 0 contributed


def test_aupimo_all_normal_batch_is_nan() -> None:
    size = 16
    maps = np.zeros((2, size, size), dtype=np.float32)
    mask = np.zeros((2, size, size), dtype=bool)
    mean_score, per_image = compute_aupimo(maps, mask)
    assert mean_score != mean_score  # NaN
    assert len(per_image) == 0


def test_aupimo_matches_pixel_metrics_field() -> None:
    """compute_pixel_metrics wires aupimo through -- confirms the integration
    point, not just the standalone function."""
    from training.evaluate import compute_pixel_metrics

    maps, mask = _step_function_case()
    pixel_metrics = compute_pixel_metrics(maps, mask)
    expected_mean, _ = compute_aupimo(maps, mask)
    assert pixel_metrics.aupimo == pytest.approx(expected_mean)
