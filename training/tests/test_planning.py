"""build_run_plan's filtering logic -- which (method, regime, dataset, seed) combos run."""

from __future__ import annotations

from pathlib import Path

from training.data import DatasetConfig, loco_defect_kind
from training.models import MethodSpec
from training.sweep import FEWSHOT_SHOTS, build_run_plan


def test_loco_defect_kind_good():
    assert loco_defect_kind(Path("cat/test/good/000.png")) == "good"


def test_loco_defect_kind_structural():
    assert loco_defect_kind(Path("cat/test/structural_anomalies/000.png")) == "structural"


def test_loco_defect_kind_logical():
    assert loco_defect_kind(Path("cat/test/logical_anomalies/000.png")) == "logical"


def _fake_fit(*_args, **_kwargs):  # pragma: no cover - never actually called
    raise NotImplementedError


def _method(name: str, trainable: bool = False, allowed_regimes=None) -> MethodSpec:
    return MethodSpec(
        name=name,
        family=name,
        backend="native",
        fit=_fake_fit,
        trainable=trainable,
        allowed_regimes=allowed_regimes,
    )


def _config(dataset: str, category: str | None) -> DatasetConfig:
    return DatasetConfig(dataset=dataset, category=category, height=256, width=256, position_aligned=True)


BOTTLE = _config("mvtec", "bottle")
HAZELNUT = _config("mvtec", "hazelnut")
CAPSULES = _config("visa", "capsules")
KOLEKTOR = _config("kolektor", None)


def test_dinomaly_excluded_from_oneclass_included_in_multiclass():
    dinomaly = _method("dinomaly_vitb14", trainable=True, allowed_regimes=("multiclass",))
    jobs = build_run_plan([dinomaly], [BOTTLE, HAZELNUT], ["oneclass", "multiclass"], seeds=(1,))

    oneclass_jobs = [j for j in jobs if j.regime == "oneclass"]
    multiclass_jobs = [j for j in jobs if j.regime == "multiclass"]
    assert oneclass_jobs == []
    assert len(multiclass_jobs) == 1


def test_unrestricted_method_runs_both_regimes():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [BOTTLE, HAZELNUT], ["oneclass", "multiclass"], seeds=(1,))
    assert len([j for j in jobs if j.regime == "oneclass"]) == 2  # one per config
    assert len([j for j in jobs if j.regime == "multiclass"]) == 1  # one per family


def test_multiclass_skips_single_category_family():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [KOLEKTOR], ["multiclass"], seeds=(1,))
    assert jobs == []


def test_fewshot_excludes_trainable_methods():
    dinomaly = _method("dinomaly_vitb14", trainable=True, allowed_regimes=("multiclass",))
    jobs = build_run_plan([dinomaly], [BOTTLE], ["fewshot"], seeds=(1, 2, 3))
    assert jobs == []


def test_fewshot_excludes_non_mvtec_visa_datasets():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [KOLEKTOR], ["fewshot"], seeds=(1,))
    assert jobs == []


def test_fewshot_includes_training_free_on_mvtec_and_visa():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [BOTTLE, CAPSULES, KOLEKTOR], ["fewshot"], seeds=(1,))
    assert len(jobs) == 2  # bottle + capsules, not kolektor


def test_fewshot_always_uses_min_seed_regardless_of_seed_count():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [BOTTLE], ["fewshot"], seeds=(3, 1, 2))
    assert len(jobs) == 1
    assert jobs[0].seed == 1


def test_oneclass_runs_once_per_seed():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [BOTTLE], ["oneclass"], seeds=(1, 2, 3))
    assert len(jobs) == 3
    assert sorted(j.seed for j in jobs) == [1, 2, 3]


def test_fitjob_result_specs_and_identity_multiclass():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [BOTTLE, HAZELNUT], ["multiclass"], seeds=(1,))
    job = jobs[0]
    specs = job.result_specs()
    assert {s.config_key for s in specs} == {"mvtec/bottle", "mvtec/hazelnut"}
    assert len({s.run_id for s in specs}) == 2  # distinct run_ids per category

    identity = job.identity_for(specs[0])
    assert identity["dataset"] == "mvtec"
    assert identity["category"] in ("bottle", "hazelnut")
    # Every column store.RunRow requires NOT NULL must be present -- this
    # caught a real bug (identity_for once omitted "regime").
    for required in ("regime", "method", "family", "backend", "config", "seed"):
        assert required in identity, f"identity_for is missing {required!r}"


def test_severstal_target_prevalence_changes_run_id_only_for_severstal():
    """A real bug this caught: run_id must be computed identically by the
    planner (before a job runs) and by regimes.py (after it runs), or
    start_run/finish_run's run_ids diverge and finish_run raises. The
    planner must only fold severstal_target_prevalence into the hash for a
    Severstal config -- never for others, or toggling the flag would
    spuriously invalidate every other dataset's resumability."""
    patchcore = _method("patchcore_resnet18")
    severstal = _config("severstal", None)

    jobs_default = build_run_plan([patchcore], [BOTTLE, severstal], ["oneclass"], seeds=(1,))
    jobs_target = build_run_plan(
        [patchcore], [BOTTLE, severstal], ["oneclass"], seeds=(1,), severstal_target_prevalence=0.275
    )

    def run_id_for(jobs, config_key):
        (job,) = [j for j in jobs if j.target.key == config_key]
        (spec,) = job.result_specs()
        return spec.run_id

    assert run_id_for(jobs_default, "severstal") != run_id_for(jobs_target, "severstal")
    assert run_id_for(jobs_default, "mvtec/bottle") == run_id_for(jobs_target, "mvtec/bottle")


def test_fitjob_result_specs_fewshot_covers_all_shots():
    patchcore = _method("patchcore_resnet18")
    jobs = build_run_plan([patchcore], [BOTTLE], ["fewshot"], seeds=(1,))
    specs = jobs[0].result_specs()
    assert len(specs) == len(FEWSHOT_SHOTS)
    assert {s.regime for s in specs} == {f"fewshot{k}" for k in FEWSHOT_SHOTS}
