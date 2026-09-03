"""The sweep: which (method, regime, dataset, seed) combinations run, and how each one runs.

Ranking methods by accuracy alone answers the wrong question. What a line
actually needs to know is how a method behaves under the constraint it will
be deployed with, and there are three regimes that matter:

* **one-class** -- one model per category. The setting almost every paper
  reports, and the easiest to score well in, because each model only has to
  represent one product. It also means 29 checkpoints to version, deploy,
  recalibrate and monitor.
* **multi-class** -- one model for an entire dataset family. One checkpoint,
  one deployment, one calibration.
* **few-shot** -- fit on a handful of normal images. The cold-start case: a
  new product arrives and nobody has collected a thousand good samples yet.

Every regime funnels into the same evaluation, so a number from one is
directly comparable to a number from another.

Organized in four parts:

1. Planning (:func:`build_run_plan` and friends) -- which combinations run,
   independent of how they're executed. Both this module's own CLI and
   anything else driving a sweep call into these functions, so a restriction
   (e.g. keeping Dinomaly out of one-class) can't be bypassed from one path
   and not another.
2. Regime execution (:func:`run_oneclass`/:func:`run_multiclass`/
   :func:`run_fewshot`) -- how one planned fit actually runs and scores.
3. Durable progress logging for an unattended multi-hour run.
4. The CLI itself (:func:`main`), which turns a selection into a plan, then
   executes it against the SQLite store, resumable at the individual-run level.

Usage:
    python -m training.sweep --list
    python -m training.sweep --regimes oneclass multiclass --models all
    python -m training.sweep --regimes multiclass --models dinomaly
    python -m training.sweep --models patchcore --datasets mvtec
    python -m training.sweep --dry-run
    python -m training.sweep --only method=patchcore dataset=mvtec_loco
"""

from __future__ import annotations

import argparse
import logging
import platform
import random
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training import store
from training.data import (
    DatasetConfig,
    discover_configs,
    has_masks,
    load_mask,
    load_split,
    loco_defect_kind,
    pixel_sample_indices,
    subsample_fit,
    subsample_test,
    subsample_to_prevalence,
)
from training.evaluate import compute_metrics, compute_pixel_metrics
from training.models import MethodSpec, RunOptions, Scorer, all_methods, scorer_labels, select

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"
DEFAULT_ARTIFACTS = RESULTS_DIR / "artifacts"
DEFAULT_CHECKPOINTS = RESULTS_DIR / "checkpoints"
DEFAULT_LOG = REPO_ROOT / "training" / "logs" / "run_all.log"

#: Regimes run when ``--regimes`` is not given. One-class is the literature
#: baseline; multi-class is the deployable result the report leads with.
DEFAULT_REGIMES = ("oneclass", "multiclass")

# -----------------------------------------------------------------------------
# Planning: which (method, regime, dataset, seed) combinations run
# -----------------------------------------------------------------------------
#
# Kept separate from the execution functions below because these are
# sweep-planning rules, not properties of a method in isolation --
# regime_allowed reads a MethodSpec's allowed_regimes override,
# allowed_for_fewshot combines a method's trainability with a dataset's
# family, and build_run_plan expands a selection into concrete units of work
# with their SQLite identity already computed.
#
# A FitJob is one GPU fit -- it may fan out into several result rows
# (multi-class fits once and scores every category; few-shot fits once per
# shot). Its expected_run_ids() is known *before* the fit runs, purely from
# the job's shape, which is what lets main() pre-register every row a job
# will produce as status="running" before calling into the regime runners --
# a crash mid-job then leaves exactly the un-produced rows detectable as
# crash debris (see training.store.reset_incomplete).

#: Shot counts swept in the few-shot regime.
FEWSHOT_SHOTS = (1, 2, 4, 8, 16)

#: Dataset families the few-shot pass touches. LOCO/Severstal/KolektorSDD2 are
#: deliberately excluded per the spec -- the cold-start question the pass
#: exists to answer is about product variety (MVTec/VisA), not defect rarity.
_FEWSHOT_DATASETS = frozenset({"mvtec", "visa"})


def regime_allowed(spec: MethodSpec, regime: str) -> bool:
    """Whether ``spec`` may run under ``regime``.

    ``spec.allowed_regimes is None`` means no restriction (the common case).
    A non-``None`` tuple is a narrow, explicit override -- today used only to
    keep Dinomaly out of the one-class regime.
    """
    return spec.allowed_regimes is None or regime in spec.allowed_regimes


def allowed_for_fewshot(spec: MethodSpec, config: DatasetConfig) -> bool:
    """Whether ``spec`` runs the few-shot pass on ``config``.

    Restricted to training-free methods (a gradient-trained model on 1-16
    images is not a meaningful few-shot result) on MVTec/VisA only.
    """
    return not spec.trainable and config.dataset in _FEWSHOT_DATASETS


@dataclass(frozen=True)
class RunSpec:
    """One result row's identity."""

    run_id: str
    method: str
    regime: str
    config_key: str
    seed: int
    defect_kind: str | None = None


@dataclass(frozen=True)
class FitJob:
    """One fit, which may score into one or many result rows.

    Attributes:
        regime: ``"oneclass"``, ``"multiclass"``, or ``"fewshot"`` (not
            shot-specific -- the shot fan-out is internal, see
            ``expected_run_ids``).
        spec: The method to fit.
        target: A single ``DatasetConfig`` for one-class/few-shot, or the
            list of every config in a family for multi-class.
        seed: The seed this fit runs under. For few-shot this is always
            ``min(seeds)`` -- see the module docstring on single-seed few-shot.
        severstal_target_prevalence: The sweep-wide
            ``--severstal-target-prevalence`` value, or ``None``. Only
            actually affects a run_id for a Severstal config (see
            ``_prevalence_for``) -- carried on every job regardless so
            ``build_run_plan`` doesn't need per-job branching, but excluded
            from the hash for every other dataset so toggling this flag
            can't spuriously change non-Severstal run_ids and break their
            resumability.
    """

    regime: str
    spec: MethodSpec
    target: DatasetConfig | list[DatasetConfig]
    seed: int
    severstal_target_prevalence: float | None = None

    def _prevalence_for(self, config: DatasetConfig) -> float | None:
        """The run_id-hashed prevalence for one config: only ever non-None
        for Severstal, matching ``_apply_severstal_target``/``_base_row`` below.
        """
        return self.severstal_target_prevalence if config.dataset == "severstal" else None

    def result_specs(self) -> list[RunSpec]:
        """Every result row this job is expected to produce, before running it."""
        if self.regime == "multiclass":
            return [
                RunSpec(
                    run_id=store.compute_run_id(
                        self.spec.name,
                        "multiclass",
                        config.key,
                        self.seed,
                        severstal_target_prevalence=self._prevalence_for(config),
                    ),
                    method=self.spec.name,
                    regime="multiclass",
                    config_key=config.key,
                    seed=self.seed,
                )
                for config in self.target
            ]
        if self.regime == "fewshot":
            config = self.target
            return [
                RunSpec(
                    run_id=store.compute_run_id(
                        self.spec.name,
                        f"fewshot{shot}",
                        config.key,
                        self.seed,
                        severstal_target_prevalence=self._prevalence_for(config),
                    ),
                    method=self.spec.name,
                    regime=f"fewshot{shot}",
                    config_key=config.key,
                    seed=self.seed,
                )
                for shot in FEWSHOT_SHOTS
            ]
        config = self.target
        return [
            RunSpec(
                run_id=store.compute_run_id(
                    self.spec.name,
                    "oneclass",
                    config.key,
                    self.seed,
                    severstal_target_prevalence=self._prevalence_for(config),
                ),
                method=self.spec.name,
                regime="oneclass",
                config_key=config.key,
                seed=self.seed,
            )
        ]

    def identity_for(self, run_spec: RunSpec) -> dict[str, Any]:
        """The identity columns to pre-register a ``run_spec`` with.

        Looks up the individual ``DatasetConfig`` the row belongs to (the
        single target for one-class/few-shot, or the matching category for
        multi-class) so ``dataset``/``category``/``height``/``width`` are
        always the specific config's, never the family's.
        """
        config = self._config_for(run_spec.config_key)
        return {
            "regime": run_spec.regime,
            "method": self.spec.name,
            "family": self.spec.family,
            "backend": self.spec.backend,
            "config": config.key,
            "dataset": config.dataset,
            "category": config.category,
            "height": config.height,
            "width": config.width,
            "seed": run_spec.seed,
        }

    def _config_for(self, config_key: str) -> DatasetConfig:
        if isinstance(self.target, list):
            for config in self.target:
                if config.key == config_key:
                    return config
            msg = f"{config_key!r} is not a category of this multiclass job"
            raise ValueError(msg)
        return self.target


def build_run_plan(
    methods: list[MethodSpec],
    configs: list[DatasetConfig],
    regimes: list[str],
    seeds: tuple[int, ...] = (1, 2, 3),
    severstal_target_prevalence: float | None = None,
) -> list[FitJob]:
    """Expand a selection into every fit that should run.

    Applies, in order: ``regime_allowed`` (keeps Dinomaly out of one-class),
    ``allowed_for_fewshot`` (training-free + MVTec/VisA only), and the
    single-seed rule for few-shot.

    Args:
        methods: Selected methods.
        configs: Selected dataset configurations.
        regimes: Regimes to plan (``"oneclass"``, ``"multiclass"``,
            ``"fewshot"``).
        seeds: Seeds to run for one-class/multi-class. Few-shot always uses
            only ``min(seeds)``.
        severstal_target_prevalence: Passed through to every job so its
            run_ids are computed consistently with what execution actually
            does at run time -- see ``FitJob._prevalence_for``.

    Returns:
        Every :class:`FitJob` the plan calls for.
    """
    jobs: list[FitJob] = []
    for regime in regimes:
        for spec in methods:
            if not regime_allowed(spec, regime):
                continue
            if regime == "multiclass":
                families: dict[str, list[DatasetConfig]] = {}
                for config in configs:
                    families.setdefault(config.dataset, []).append(config)
                for family_configs in families.values():
                    # A single-category family has no multi-class question to
                    # answer; it would just duplicate the one-class row.
                    if len(family_configs) <= 1:
                        continue
                    for seed in seeds:
                        jobs.append(
                            FitJob(
                                "multiclass", spec, family_configs, seed, severstal_target_prevalence
                            )
                        )
            elif regime == "fewshot":
                for config in configs:
                    if allowed_for_fewshot(spec, config):
                        jobs.append(
                            FitJob(
                                "fewshot", spec, config, min(seeds), severstal_target_prevalence
                            )
                        )
            else:
                for config in configs:
                    for seed in seeds:
                        jobs.append(
                            FitJob("oneclass", spec, config, seed, severstal_target_prevalence)
                        )
    return jobs


# -----------------------------------------------------------------------------
# Regime execution: how one planned fit actually runs and scores
# -----------------------------------------------------------------------------


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


def _apply_severstal_target(
    config: DatasetConfig, test_split: list[tuple[Path, bool]], options: RunOptions
) -> list[tuple[Path, bool]]:
    """Subsample Severstal's test split to ``options.severstal_target_prevalence``, if set.

    A no-op for every other dataset and when the option is unset (``None``,
    the default) -- restricting cross-dataset AP/F1 comparison
    (``training.evaluate``'s leaderboard) is the default fix; this is the
    opt-in alternative.
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
        # and may be contaminated (see training.data's severstal_paths).
        test_prevalence=round(float(labels.mean()), 4) if len(labels) else float("nan"),
        label_noise_caveat=config.dataset == "severstal",
        severstal_target_prevalence=context.options.severstal_target_prevalence,
        **compute_metrics(scores, labels).as_dict(),
    )

    if maps is not None and has_masks(config):
        pixel = _pixel_metrics(maps, test_split, config, context)
        if pixel is not None:
            row.update(pixel.as_dict())

    store.save_artifact(
        store.artifact_path(context.artifact_root, regime, spec.name, config.key, context.options.seed),
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
    combined row's by filtering with ``training.data.loco_defect_kind`` if
    ever needed, so nothing is lost by not persisting it separately.

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
    indices = pixel_sample_indices(test_split, context.options.seed)
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
        store.save_checkpoint(
            scorer, store.checkpoint_path(context.checkpoint_root, "oneclass", spec.name, config.key)
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
        store.save_checkpoint(
            scorer, store.checkpoint_path(context.checkpoint_root, "multiclass", spec.name, family)
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
            store.save_checkpoint(
                scorer,
                store.checkpoint_path(context.checkpoint_root, f"fewshot{shot}", spec.name, config.key),
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
                    # Always 0.0 in practice -- allowed_for_fewshot above
                    # restricts this regime to training-free methods -- but
                    # set explicitly for schema consistency with the other
                    # regimes.
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
#: of configs rather than one, so the CLI dispatches it separately.
REGIME_RUNNERS: dict[str, Callable[..., Iterator[dict[str, Any]]]] = {
    "oneclass": run_oneclass,
    "multiclass": run_multiclass,
    "fewshot": run_fewshot,
}


# -----------------------------------------------------------------------------
# Durable progress logging for an unattended multi-hour sweep
# -----------------------------------------------------------------------------
#
# The CLI below already prints progress to stdout, which is enough for a
# terminal someone is watching. This adds the same lines to a rotating file
# too, so progress since the last reboot/relaunch is inspectable without
# having kept a terminal open (``tail -f training/logs/run_all.log``).

_LOGGER_NAME = "benchmark"


def configure_logging(log_path: Path) -> logging.Logger:
    """Attach a rotating file handler (and, if absent, a stdout handler).

    Idempotent -- safe to call from both this module's CLI and
    ``training.train``'s ``all`` pipeline when one launches the other as a
    subprocess pointed at the same file.

    Args:
        log_path: Destination log file. Created, along with its parent
            directory, if absent.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path.resolve())
        for h in logger.handlers
    )
    if not has_file_handler:
        file_handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(file_handler)

    # RotatingFileHandler is itself a StreamHandler subclass, so this must
    # exclude file handlers explicitly or it never adds the console one.
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


# -----------------------------------------------------------------------------
# CLI: turn a selection into a plan, then execute it
# -----------------------------------------------------------------------------

#: Rough per-job duration guesses, used for --dry-run's ETA only when this
#: DB has no historical timing for the method in question yet. Training-free
#: fits are seconds; gradient-trained (Dinomaly) fits are the real cost.
_FALLBACK_SECONDS_TRAINABLE = 45 * 60
_FALLBACK_SECONDS_TRAINING_FREE = 30


def select_configs(selectors: list[str], data_root: Path) -> list[DatasetConfig]:
    """Resolve ``--datasets`` selectors into concrete configurations.

    A selector is ``all``, a dataset family (``mvtec``), or a full key
    (``mvtec/bottle``).

    Raises:
        SystemExit: If a selector matches nothing on disk.
    """
    available = discover_configs(data_root)
    if not available:
        msg = f"No datasets found under {data_root}. Pass --data-root."
        raise SystemExit(msg)

    chosen: dict[str, DatasetConfig] = {}
    for selector in selectors:
        if selector == "all":
            chosen.update({config.key: config for config in available})
            continue
        matches = [c for c in available if c.key == selector or c.dataset == selector]
        if not matches:
            keys = ", ".join(sorted({c.dataset for c in available}))
            msg = f"Selector {selector!r} matched no dataset. Families present: {keys}"
            raise SystemExit(msg)
        chosen.update({config.key: config for config in matches})
    return sorted(chosen.values(), key=lambda config: config.key)


def _parse_only(tokens: list[str]) -> dict[str, set[str]]:
    """Parse ``--only key=value`` tokens into ``{key: {value, ...}}``.

    Same key repeated is OR'd together; different keys are AND'd.
    """
    parsed: dict[str, set[str]] = {}
    for token in tokens:
        if "=" not in token:
            msg = f"--only expects key=value tokens, got {token!r}"
            raise SystemExit(msg)
        key, value = token.split("=", 1)
        parsed.setdefault(key, set()).add(value)
    return parsed


def _job_matches_only(job: FitJob, only: dict[str, set[str]]) -> bool:
    """Whether ``job`` satisfies every ``--only`` constraint."""
    if "method" in only and not (
        job.spec.name in only["method"]
        or job.spec.family in only["method"]
        or job.spec.backend in only["method"]
    ):
        return False
    if "dataset" in only:
        configs = job.target if isinstance(job.target, list) else [job.target]
        if not any(c.dataset in only["dataset"] or c.key in only["dataset"] for c in configs):
            return False
    if "regime" in only and job.regime not in only["regime"]:
        return False
    return True


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch so a rerun reproduces the same row."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _describe(row: dict[str, Any]) -> str:
    """One-line summary of a finished run."""
    if row["status"] != "ok":
        return f"FAILED {row['error']}"
    parts = [
        f"AUROC={row['auroc']:.4f}",
        f"AP={row['average_precision']:.4f}",
        f"F1={row['f1_max']:.4f}",
        f"scrap@95={row['fpr_at_95tpr']:.3f}",
    ]
    if "aupro" in row and row["aupro"] == row["aupro"]:  # present and not NaN
        parts.append(f"AUPRO={row['aupro']:.4f}")
    parts.append(f"{row['ms_per_image']:.1f}ms/img")
    return " ".join(parts)


def job_label(job: FitJob) -> str:
    """Human-readable name for a unit of work."""
    if job.regime == "multiclass":
        return f"{job.spec.name} @ {job.target[0].dataset} (x{len(job.target)} categories) seed={job.seed}"
    return f"{job.spec.name} @ {job.target.key} seed={job.seed}"


def execute(job: FitJob, context: RunContext):
    """Dispatch one job to its regime runner."""
    if job.regime == "multiclass":
        return run_multiclass(job.spec, job.target, context)
    if job.regime == "fewshot":
        return run_fewshot(job.spec, job.target, context)
    return run_oneclass(job.spec, job.target, context)


def print_zoo() -> None:
    """Print the registered zoo, grouped by family."""
    methods = all_methods()
    print(f"{len(methods)} methods registered\n")
    family = None
    for spec in sorted(methods, key=lambda s: (s.family, s.name)):
        if spec.family != family:
            family = spec.family
            print(f"[{family}]")
        export = "onnx" if spec.exportable else "    "
        print(f"  {export}  {spec.name:38s} {spec.notes}")
    if not any(spec.backend == "anomalib" for spec in methods):
        print("\n(Anomalib not installed -- `pip install anomalib` adds its zoo.)")


def _estimate_seconds(session_factory: Any, pending: list[FitJob]) -> float:
    """Blend real historical per-job durations with a rough fallback guess."""
    from sqlalchemy import select as sa_select

    from training.store import RunRow, session_scope

    history: dict[str, list[float]] = {}
    with session_scope(session_factory) as session:
        rows = session.scalars(
            sa_select(RunRow).where(
                RunRow.status == "ok", RunRow.fit_seconds.is_not(None), RunRow.score_seconds.is_not(None)
            )
        )
        for row in rows:
            history.setdefault(row.method, []).append((row.fit_seconds or 0) + (row.score_seconds or 0))

    global_history = [seconds for seconds_list in history.values() for seconds in seconds_list]
    global_average = sum(global_history) / len(global_history) if global_history else None

    total = 0.0
    for job in pending:
        durations = history.get(job.spec.name)
        if durations:
            total += sum(durations) / len(durations)
        elif global_average is not None:
            total += global_average
        else:
            total += (
                _FALLBACK_SECONDS_TRAINABLE if job.spec.trainable else _FALLBACK_SECONDS_TRAINING_FREE
            )
    return total


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the sweep."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--models", nargs="+", default=["all"], help="Names, families, or 'all'.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Keys, families, or 'all'.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(DEFAULT_REGIMES),
        choices=["oneclass", "multiclass", "fewshot"],
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-fit-images", type=int, default=500, help="0 uses every image.")
    parser.add_argument("--max-test-images", type=int, default=0, help="0 uses the whole split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs/iterations.")
    parser.add_argument(
        "--severstal-target-prevalence",
        type=float,
        default=None,
        help="Downsample Severstal's test split to this anomalous-image rate.",
    )
    parser.add_argument("--no-pixel", action="store_true", help="Skip localization metrics.")
    parser.add_argument("--force", action="store_true", help="Re-run cells that already passed.")
    parser.add_argument("--list", action="store_true", help="Print the zoo and exit.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and an ETA; run nothing."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Filter the plan, e.g. --only method=patchcore dataset=mvtec_loco",
    )
    args = parser.parse_args(argv)

    if args.list:
        print_zoo()
        return

    logger = configure_logging(args.log_file)

    methods = select(args.models)
    configs = select_configs(args.datasets, args.data_root)
    plan = build_run_plan(
        methods,
        configs,
        args.regimes,
        seeds=tuple(args.seeds),
        severstal_target_prevalence=args.severstal_target_prevalence,
    )

    only = _parse_only(args.only)
    if only:
        plan = [job for job in plan if _job_matches_only(job, only)]

    _, session_factory = store.open_database(args.results_db)

    scope_ids = {result_spec.run_id for job in plan for result_spec in job.result_specs()}

    if not args.dry_run:
        deleted = store.reset_incomplete(session_factory, scope_ids, force=args.force)
        if deleted:
            logger.info("cleared %d incomplete/failed row(s) from a prior run", deleted)

    done_ids = set() if args.force else store.completed_run_ids(session_factory, scope_ids)
    pending = [job for job in plan if not all(rs.run_id in done_ids for rs in job.result_specs())]

    total_rows = len(scope_ids)
    only_note = f" (--only narrowed {len(methods)}x{len(configs)} selection)" if only else ""
    logger.info(
        "%d fits planned%s: %d regimes x %d seeds = %d rows total (%d already done)",
        len(plan),
        only_note,
        len(args.regimes),
        len(args.seeds),
        total_rows,
        len(done_ids),
    )
    logger.info("device=%s pixel_metrics=%s", args.device, not args.no_pixel)
    logger.info("results_db=%s", args.results_db)
    logger.info("host=%s torch=%s", platform.node(), torch.__version__)

    if args.dry_run:
        estimated_seconds = _estimate_seconds(session_factory, pending)
        logger.info(
            "\n--dry-run: %d fits pending, estimated %.1f min total. Nothing executed.",
            len(pending),
            estimated_seconds / 60,
        )
        return

    options = RunOptions(
        device=args.device,
        max_fit_images=args.max_fit_images,
        max_test_images=args.max_test_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seeds[0],
        epochs=args.epochs,
        severstal_target_prevalence=args.severstal_target_prevalence,
    )
    context = RunContext(
        args.data_root, options, args.artifacts, args.checkpoints, want_pixel=not args.no_pixel
    )

    started = time.perf_counter()
    elapsed_per_job: list[float] = []
    for index, job in enumerate(pending, start=1):
        options.seed = job.seed
        for result_spec in job.result_specs():
            store.start_run(session_factory, result_spec.run_id, job.identity_for(result_spec))

        remaining = len(pending) - index + 1
        if elapsed_per_job:
            avg = sum(elapsed_per_job) / len(elapsed_per_job)
            eta = f"ETA {avg * remaining / 60:.1f} min (avg {avg:.1f}s/fit)"
        else:
            eta = "ETA unknown (first fit)"
        logger.info("[%d/%d] %s: %s ... %s", index, len(pending), job.regime, job_label(job), eta)

        seed_everything(job.seed)
        job_start = time.perf_counter()
        for row in execute(job, context):
            row["elapsed_total_s"] = round(time.perf_counter() - started, 1)
            run_id = store.compute_run_id(
                row["method"],
                row["regime"],
                row["config"],
                row["seed"],
                row.get("defect_kind"),
                row.get("severstal_target_prevalence"),
            )
            store.finish_run(session_factory, run_id, row)
            logger.info("    %s: %s", row["config"], _describe(row))
        elapsed_per_job.append(time.perf_counter() - job_start)

    logger.info("\nDone in %.1f min -> %s", (time.perf_counter() - started) / 60, args.results_db)
    logger.info("Now run: python -m training.evaluate leaderboard")


if __name__ == "__main__":
    main()
