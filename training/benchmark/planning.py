"""Shared policy for which (method, regime, dataset, seed) combinations run.

Kept separate from ``registry.py`` because these are sweep-planning rules, not
properties of a method in isolation -- ``regime_allowed`` reads a
``MethodSpec``'s ``allowed_regimes`` override, ``allowed_for_fewshot``
combines a method's trainability with a dataset's family, and
``build_run_plan`` expands a selection into concrete units of work with
their SQLite identity already computed. Both the legacy direct-CLI path
(``run.py``) and anything else driving a sweep call into this module, so a
restriction can't be bypassed from one path and not the other.

A :class:`FitJob` is one GPU fit -- it may fan out into several result rows
(multi-class fits once and scores every category; few-shot fits once per
shot). Its ``expected_run_ids()`` is known *before* the fit runs, purely from
the job's shape, which is what lets the orchestrator pre-register every row a
job will produce as ``status="running"`` before calling into ``regimes.py`` --
a crash mid-job then leaves exactly the un-produced rows detectable as
crash debris (see ``store.reset_incomplete``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.data import DatasetConfig
from benchmark.registry import MethodSpec
from benchmark.store import compute_run_id

#: Shot counts swept in the few-shot regime. Duplicated from regimes.py's
#: FEWSHOT_SHOTS (not imported) to keep this module import-independent of
#: regimes.py -- planning describes what runs, regimes.py how to run it.
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
        for Severstal, matching regimes.py's _apply_severstal_target/_base_row.
        """
        return self.severstal_target_prevalence if config.dataset == "severstal" else None

    def result_specs(self) -> list[RunSpec]:
        """Every result row this job is expected to produce, before running it."""
        if self.regime == "multiclass":
            return [
                RunSpec(
                    run_id=compute_run_id(
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
                    run_id=compute_run_id(
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
                run_id=compute_run_id(
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

    Applies, in order: §1's ``regime_allowed`` (keeps Dinomaly out of
    one-class), §7's ``allowed_for_fewshot`` (training-free + MVTec/VisA
    only), and the single-seed rule for few-shot.

    Args:
        methods: Selected methods.
        configs: Selected dataset configurations.
        regimes: Regimes to plan (``"oneclass"``, ``"multiclass"``,
            ``"fewshot"``).
        seeds: Seeds to run for one-class/multi-class. Few-shot always uses
            only ``min(seeds)``.
        severstal_target_prevalence: Passed through to every job so its
            run_ids are computed consistently with what ``regimes.py``
            actually does at execution time -- see ``FitJob._prevalence_for``.

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
