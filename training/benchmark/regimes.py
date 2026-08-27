"""The three deployment regimes a method is measured under.

Ranking methods by accuracy alone answers the wrong question. What a line
actually needs to know is how a method behaves under the constraint it will be
deployed with, and there are three that matter:

* **one-class** -- one model per category. The setting almost every paper
  reports, and the easiest to score well in, because each model only has to
  represent one product. It also means 29 checkpoints to version, deploy,
  recalibrate and monitor.
* **multi-class** -- one model for an entire dataset family. One checkpoint,
  one deployment, one calibration. Until recently this cost several points of
  AUROC; the reconstruction models of 2025 largely closed that gap, and
  whether they close it *here* is the central question of this benchmark.
* **few-shot** -- fit on a handful of normal images. This is the cold-start
  case: a new product arrives and nobody has collected a thousand good
  samples yet. A method that needs 500 images to work is a method that cannot
  be switched on for two weeks.

Every regime funnels into the same evaluation, so a number from one is
directly comparable to a number from another.
"""

from __future__ import annotations

import random
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets import subsample_to_prevalence

from benchmark.artifacts import artifact_path, save_artifact
from benchmark.checkpoints import checkpoint_path, save_checkpoint
from benchmark.data import DatasetConfig, load_split, loco_defect_kind
from benchmark.evaluation import compute_metrics, compute_pixel_metrics
from benchmark.masks import has_masks, load_mask
from benchmark.registry import MethodSpec, RunOptions, Scorer, scorer_labels

#: Shot counts swept in the few-shot regime.
FEWSHOT_SHOTS = (1, 2, 4, 8, 16)

#: Cap on images whose masks are loaded for pixel metrics. Decoding and
#: resizing masks for all ~7,250 Severstal test images, once per method, would
#: dominate the sweep; a class-balanced sample estimates the same quantity.
_MAX_PIXEL_IMAGES = 400


class RunContext:
    """Everything a regime needs that is not the method or the config.

    Args:
        data_root: Directory holding the dataset corpora.
        options: Sweep-wide knobs.
        artifact_root: Where per-run score/map archives are written.
        checkpoint_root: Where per-run fitted models are written.
        want_pixel: Whether to compute localization metrics.
    """

    def __init__(
        self,
        data_root: Path,
        options: RunOptions,
        artifact_root: Path,
        checkpoint_root: Path,
        want_pixel: bool = True,
    ) -> None:
        """Store the context."""
        self.data_root = data_root
        self.options = options
        self.artifact_root = artifact_root
        self.checkpoint_root = checkpoint_root
        self.want_pixel = want_pixel


def subsample_fit(paths: list[Path], limit: int, seed: int) -> list[Path]:
    """Take a seeded random subset of the normal training images.

    Random rather than head-of-list: several corpora are ordered by capture
    session, so the first N images can share a lighting or batch artifact that
    the rest of the set does not.
    """
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def subsample_test(
    split: list[tuple[Path, bool]], limit: int, seed: int
) -> list[tuple[Path, bool]]:
    """Take a class-balanced seeded subset of the test split.

    Capping without balancing would silently change the positive rate, which
    moves average precision and F1 and makes configs incomparable.
    """
    if limit <= 0 or len(split) <= limit:
        return list(split)
    rng = random.Random(seed)
    normal = [item for item in split if not item[1]]
    anomalous = [item for item in split if item[1]]
    rng.shuffle(normal)
    rng.shuffle(anomalous)
    per_class = max(1, limit // 2)
    kept = normal[:per_class] + anomalous[:per_class]
    rng.shuffle(kept)
    return kept


def _apply_severstal_target(
    config: DatasetConfig, test_split: list[tuple[Path, bool]], options: RunOptions
) -> list[tuple[Path, bool]]:
    """Subsample Severstal's test split to ``options.severstal_target_prevalence``, if set.

    A no-op for every other dataset and when the option is unset (``None``,
    the default) -- restricting cross-dataset AP/F1 comparison
    (``leaderboard.py``) is the default fix; this is the opt-in alternative.
    """
    if config.dataset != "severstal" or options.severstal_target_prevalence is None:
        return test_split
    return subsample_to_prevalence(test_split, options.severstal_target_prevalence, options.seed)


def _base_row(spec: MethodSpec, config: DatasetConfig, regime: str, options: RunOptions) -> dict:
    """The identifying columns every result row carries."""
    return {
        "regime": regime,
        "method": spec.name,
        "family": spec.family,
        "backend": spec.backend,
        "config": config.key,
        "dataset": config.dataset,
        "category": config.category,
        "height": config.height,
        "width": config.width,
        "seed": options.seed,
    }


def evaluate_scorer(
    scorer: Scorer,
    spec: MethodSpec,
    config: DatasetConfig,
    test_split: list[tuple[Path, bool]],
    context: RunContext,
    regime: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a fitted method on one config and assemble its result row.

    Also persists the raw scores, labels and heatmaps so the dashboard and the
    ensembler never have to re-run anything.

    Args:
        scorer: The fitted method.
        spec: Its registry entry.
        config: The configuration being evaluated.
        test_split: Labeled test images.
        context: Shared run context.
        regime: Regime label recorded on the row.
        extra: Additional columns to merge in.

    Returns:
        A populated result row with ``status="ok"``.
    """
    test_paths = [path for path, _ in test_split]
    labels = np.array([is_anomalous for _, is_anomalous in test_split], dtype=bool)

    want_maps = context.want_pixel and getattr(scorer, "produces_maps", False)
    score_start = time.perf_counter()
    if hasattr(scorer, "score_with_maps"):
        scores, maps = scorer.score_with_maps(test_paths, want_maps=want_maps)
    else:
        scores, maps = scorer.score(test_paths), None
    score_seconds = time.perf_counter() - score_start

    # A backend that owns its evaluation order supplies matching labels.
    own_labels = scorer_labels(scorer)
    if own_labels is not None:
        labels = own_labels

    row = _base_row(spec, config, regime, context.options)
    row.update(
        status="ok",
        n_test=len(test_paths),
        score_seconds=round(score_seconds, 3),
        ms_per_image=round(score_seconds / max(1, len(test_paths)) * 1000, 3),
        peak_vram_gb=round(_peak_vram_gb(), 3),
        # Stored on every row, not just Severstal's -- cheap, and useful for
        # comparing datasets in general. label_noise_caveat flags that
        # Severstal's held-out "normal" pool is Kaggle-competition-labeled
        # and may be contaminated (see datasets.py's severstal_paths).
        test_prevalence=round(float(labels.mean()), 4) if len(labels) else float("nan"),
        label_noise_caveat=config.dataset == "severstal",
        severstal_target_prevalence=context.options.severstal_target_prevalence,
        **compute_metrics(scores, labels).as_dict(),
    )

    if maps is not None and has_masks(config):
        pixel = _pixel_metrics(maps, test_split, config, context)
        if pixel is not None:
            row.update(pixel.as_dict())

    save_artifact(
        artifact_path(context.artifact_root, regime, spec.name, config.key, context.options.seed),
        scores,
        labels,
        [str(path) for path in test_paths],
        maps,
    )
    if extra:
        row.update(extra)
    return row


def evaluate_loco_breakdown(
    scorer: Scorer,
    spec: MethodSpec,
    config: DatasetConfig,
    test_split: list[tuple[Path, bool]],
    context: RunContext,
    regime: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Structural-only and logical-only rows for a LOCO config.

    Called *in addition to* ``evaluate_scorer``'s combined-mean row (see the
    ``config.dataset == "mvtec_loco"`` branch in each regime runner below) --
    together they give "a combined mean... [and] structural and logical
    defects separately" per the spec.

    Deliberately does not reuse ``evaluate_scorer`` wholesale: that function
    saves an artifact keyed only by (regime, method, config, seed), and
    calling it three times for the same LOCO config would have each
    breakdown call silently overwrite the previous one's saved scores. This
    re-scores the same already-fitted scorer (no re-fit -- scoring a subset
    of already-loaded images is cheap) and does not save its own artifact;
    a structural- or logical-only artifact is always derivable from the
    combined row's by filtering with ``data.loco_defect_kind`` if ever
    needed, so nothing is lost by not persisting it separately.

    Args:
        scorer: The fitted method -- same instance evaluate_scorer just used.
        spec: Its registry entry.
        config: The MVTec LOCO configuration being evaluated.
        test_split: The SAME full labeled test split evaluate_scorer just
            scored (this partitions it, not a fresh load).
        context: Shared run context.
        regime: Regime label recorded on each row.
        extra: Additional columns to merge in, matching evaluate_scorer's.

    Returns:
        Zero, one, or two rows (skipping a kind if this config has none of
        it), each with ``defect_kind`` set to ``"structural"`` or ``"logical"``.
    """
    rows: list[dict[str, Any]] = []
    for kind in ("structural", "logical"):
        subset = [
            item for item in test_split if not item[1] or loco_defect_kind(item[0]) == kind
        ]
        n_bad = sum(1 for _, is_anomalous in subset if is_anomalous)
        n_good = len(subset) - n_bad
        if n_good == 0 or n_bad == 0:
            continue  # nothing of this kind here; a degenerate row would be all-nan anyway

        test_paths = [path for path, _ in subset]
        labels = np.array([is_anomalous for _, is_anomalous in subset], dtype=bool)
        want_maps = context.want_pixel and getattr(scorer, "produces_maps", False)
        if hasattr(scorer, "score_with_maps"):
            scores, maps = scorer.score_with_maps(test_paths, want_maps=want_maps)
        else:
            scores, maps = scorer.score(test_paths), None

        own_labels = scorer_labels(scorer)
        if own_labels is not None:
            labels = own_labels

        row = _base_row(spec, config, regime, context.options)
        row.update(
            status="ok",
            defect_kind=kind,
            n_test=len(test_paths),
            test_prevalence=round(float(labels.mean()), 4) if len(labels) else float("nan"),
            label_noise_caveat=False,
            severstal_target_prevalence=None,
            **compute_metrics(scores, labels).as_dict(),
        )
        if maps is not None and has_masks(config):
            pixel = _pixel_metrics(maps, subset, config, context)
            if pixel is not None:
                row.update(pixel.as_dict())
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _pixel_metrics(
    maps: np.ndarray,
    test_split: list[tuple[Path, bool]],
    config: DatasetConfig,
    context: RunContext,
):
    """Compute localization metrics over a capped, mask-bearing sample.

    Masks are resized to match the map's *actual* resolution, not the
    dataset's declared config geometry -- most methods' maps do come out at
    config.height/width, but Dinomaly always runs at its own fixed
    resolution (see DinomalyScorer.input_size) regardless of the dataset.
    Requesting a mask sized to the config for a method whose maps are a
    different shape stacks mismatched arrays and roc_auc_score rejects them
    outright ("inconsistent numbers of samples") -- this was silently
    breaking every Dinomaly pixel-metrics computation.
    """
    indices = _pixel_sample_indices(test_split, context.options.seed)
    selected_maps = []
    selected_masks = []

    for index in indices:
        path = test_split[index][0]
        map_height, map_width = maps[index].shape[-2:]
        mask = load_mask(path, config, context.data_root, map_height, map_width)
        if mask is None:
            continue
        selected_maps.append(maps[index])
        selected_masks.append(mask)

    if not selected_masks:
        return None
    return compute_pixel_metrics(np.stack(selected_maps), np.stack(selected_masks))


def _pixel_sample_indices(test_split: list[tuple[Path, bool]], seed: int) -> list[int]:
    """Indices of a class-balanced sample capped at :data:`_MAX_PIXEL_IMAGES`.

    Normal images are included deliberately: a model that lights up clean
    parts should be punished on pixel AUROC, and a defect-only sample would
    hide exactly that.
    """
    if len(test_split) <= _MAX_PIXEL_IMAGES:
        return list(range(len(test_split)))
    rng = random.Random(seed)
    anomalous = [i for i, (_, is_anom) in enumerate(test_split) if is_anom]
    normal = [i for i, (_, is_anom) in enumerate(test_split) if not is_anom]
    rng.shuffle(anomalous)
    rng.shuffle(normal)
    per_class = _MAX_PIXEL_IMAGES // 2
    return sorted(anomalous[:per_class] + normal[:per_class])


def _peak_vram_gb() -> float:
    """Peak CUDA allocation since the last reset, in GB (``0`` on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _failed_row(
    spec: MethodSpec, config: DatasetConfig, regime: str, options: RunOptions, exc: Exception
) -> dict[str, Any]:
    """Record a crash as a result rather than losing it."""
    row = _base_row(spec, config, regime, options)
    row.update(
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(limit=8),
    )
    return row


def _prepare(
    config: DatasetConfig, context: RunContext
) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Load and subsample one config's split."""
    train_paths, test_split = load_split(config, context.data_root)
    options = context.options
    test_split = _apply_severstal_target(config, test_split, options)
    return (
        subsample_fit(train_paths, options.max_fit_images, options.seed),
        subsample_test(test_split, options.max_test_images, options.seed),
    )


def run_oneclass(
    spec: MethodSpec, config: DatasetConfig, context: RunContext
) -> Iterator[dict[str, Any]]:
    """Fit on one category's normal images and evaluate on that category."""
    try:
        train_paths, test_split = _prepare(config, context)
        _reset_vram()
        fit_start = time.perf_counter()
        scorer = spec.fit(config, train_paths, test_split, context.options)
        fit_seconds = time.perf_counter() - fit_start
        save_checkpoint(
            scorer, checkpoint_path(context.checkpoint_root, "oneclass", spec.name, config.key)
        )
        extra = {
            "n_fit": len(train_paths),
            "fit_seconds": round(fit_seconds, 3),
            "training_wall_clock_seconds": round(fit_seconds, 3) if spec.trainable else 0.0,
            "single_seed": False,
        }
        yield evaluate_scorer(scorer, spec, config, test_split, context, "oneclass", extra)
        if config.dataset == "mvtec_loco":
            yield from evaluate_loco_breakdown(
                scorer, spec, config, test_split, context, "oneclass", extra
            )
    except Exception as exc:
        yield _failed_row(spec, config, "oneclass", context.options, exc)
    finally:
        _release()


def run_multiclass(
    spec: MethodSpec, configs: list[DatasetConfig], context: RunContext
) -> Iterator[dict[str, Any]]:
    """Fit one model on every category in a family, then evaluate each separately.

    This is the regime that decides whether the project ships one model or
    twenty-nine. The fit sees the pooled normal images of every category with
    no category label, so nothing tells the model which product it is looking
    at -- at inference it has to be right without being told.

    Args:
        spec: The method to fit.
        configs: Every configuration in one dataset family.
        context: Shared run context.

    Yields:
        One result row per configuration, all sharing the single fit.
    """
    family = configs[0].dataset
    try:
        pooled_train: list[Path] = []
        per_config_test: list[tuple[DatasetConfig, list[tuple[Path, bool]]]] = []

        # Budget the fit set across categories so one large category cannot
        # dominate the pooled distribution.
        options = context.options
        per_category = (
            max(1, options.max_fit_images // max(1, len(configs)))
            if options.max_fit_images > 0
            else 0
        )
        for config in configs:
            train_paths, test_split = load_split(config, context.data_root)
            test_split = _apply_severstal_target(config, test_split, options)
            pooled_train.extend(subsample_fit(train_paths, per_category, options.seed))
            per_config_test.append(
                (config, subsample_test(test_split, options.max_test_images, options.seed))
            )

        _reset_vram()
        fit_start = time.perf_counter()
        # The fit is handed the first config purely for input geometry; every
        # category in a family shares it by construction.
        scorer = spec.fit(configs[0], pooled_train, [], options)
        fit_seconds = time.perf_counter() - fit_start
        # One checkpoint per family, not per category -- every category's row
        # below shares this exact fit, so saving it once avoids N redundant
        # copies of the same model.
        save_checkpoint(
            scorer, checkpoint_path(context.checkpoint_root, "multiclass", spec.name, family)
        )

        for config, test_split in per_config_test:
            extra = {
                "n_fit": len(pooled_train),
                "fit_seconds": round(fit_seconds, 3),
                "training_wall_clock_seconds": round(fit_seconds, 3) if spec.trainable else 0.0,
                "single_seed": False,
                "multiclass_family": family,
                "multiclass_categories": len(configs),
            }
            yield evaluate_scorer(scorer, spec, config, test_split, context, "multiclass", extra)
            if config.dataset == "mvtec_loco":
                yield from evaluate_loco_breakdown(
                    scorer, spec, config, test_split, context, "multiclass", extra
                )
    except Exception as exc:
        for config in configs:
            yield _failed_row(spec, config, "multiclass", context.options, exc)
    finally:
        _release()


def run_fewshot(
    spec: MethodSpec,
    config: DatasetConfig,
    context: RunContext,
    shots: tuple[int, ...] = FEWSHOT_SHOTS,
) -> Iterator[dict[str, Any]]:
    """Fit on ``k`` normal images for each ``k``, to trace the cold-start curve."""
    for shot in shots:
        try:
            train_paths, test_split = load_split(config, context.data_root)
            sampled = subsample_fit(train_paths, shot, context.options.seed)
            if len(sampled) < shot:
                continue
            test_split = subsample_test(
                test_split, context.options.max_test_images, context.options.seed
            )

            _reset_vram()
            fit_start = time.perf_counter()
            scorer = spec.fit(config, sampled, test_split, context.options)
            fit_seconds = time.perf_counter() - fit_start
            # Regime is already shot-specific ("fewshot4", "fewshot8", ...),
            # so it alone disambiguates the checkpoint path across shots.
            save_checkpoint(
                scorer,
                checkpoint_path(context.checkpoint_root, f"fewshot{shot}", spec.name, config.key),
            )
            yield evaluate_scorer(
                scorer,
                spec,
                config,
                test_split,
                context,
                f"fewshot{shot}",
                {
                    "n_fit": len(sampled),
                    "n_shot": shot,
                    "fit_seconds": round(fit_seconds, 3),
                    # Always 0.0 in practice -- allowed_for_fewshot (planning.py)
                    # restricts this regime to training-free methods -- but set
                    # explicitly for schema consistency with the other regimes.
                    "training_wall_clock_seconds": round(fit_seconds, 3) if spec.trainable else 0.0,
                    "single_seed": True,
                },
            )
        except Exception as exc:
            yield _failed_row(spec, config, f"fewshot{shot}", context.options, exc)
        finally:
            _release()


def _reset_vram() -> None:
    """Reset the CUDA peak-memory counter before a fit."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _release() -> None:
    """Return cached CUDA blocks between runs so the next fit starts clean."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: Regime name to the callable that executes it. ``multiclass`` takes a list
#: of configs rather than one, so the runner dispatches it separately.
REGIME_RUNNERS: dict[str, Callable[..., Iterator[dict[str, Any]]]] = {
    "oneclass": run_oneclass,
    "multiclass": run_multiclass,
    "fewshot": run_fewshot,
}
