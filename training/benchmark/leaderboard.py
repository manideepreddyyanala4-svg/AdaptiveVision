"""Aggregate sweep results into a ranking and a written report.

A pile of per-run AUROC numbers is not an answer to "which model should we
ship". This turns the SQLite store into the things that are:

* a ranking per regime, restricted to methods that actually completed every
  configuration -- averaging a method over the six categories it survived and
  calling that a win over one that ran all twenty-nine is the easiest way to
  pick the wrong model;
* the regime comparison, which is the real finding: whether one model can
  cover every category or whether the project ships one per category;
* per-dataset winners, because the right answer genuinely differs between
  aligned parts and continuous material;
* operating-point error rates and a latency column, because a station has a
  cycle time and has to commit to a threshold.

Every metric is reported as mean +/- std across seeds, never a bare number --
see :func:`aggregate_seeds`.

Usage:
    python training/benchmark/leaderboard.py
    python training/benchmark/leaderboard.py --results-db path/to/benchmark.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):  # Allow `python training/benchmark/leaderboard.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import store

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DB = REPO_ROOT / "training" / "benchmark_results" / "benchmark.db"

#: Columns aggregated as a mean, with the label used in the report. Cost
#: columns are aggregated the same way as accuracy ones -- the whole point of
#: capturing them per-run is that they get the same mean+/-std treatment.
_MEAN_COLUMNS = {
    "auroc": "mean_auroc",
    "average_precision": "mean_ap",
    "f1_max": "mean_f1",
    "fpr_at_95tpr": "scrap_at_95",
    "fnr_at_1fpr": "escape_at_1fpr",
    "pg2": "mean_pg2",
    "pb2": "mean_pb2",
    "balanced_error_rate": "bal_error",
    "aupro": "mean_aupro",
    "aupimo": "mean_aupimo",
    "pixel_auroc": "mean_pixel_auroc",
    "ms_per_image": "ms_per_image",
    "fit_seconds": "fit_seconds",
    "inference_latency_ms_p50": "latency_p50_ms",
    "inference_latency_ms_p95": "latency_p95_ms",
    "throughput_fps_bs1": "throughput_fps_bs1",
    "throughput_fps_bs16": "throughput_fps_bs16",
    "model_params_millions": "model_params_m",
    "peak_gpu_memory_mb": "peak_gpu_mb",
    "training_wall_clock_seconds": "train_wall_clock_s",
}

#: Metrics that are prevalence-sensitive and therefore unsafe to compare
#: across datasets with different positive rates (Severstal's ~92% positive
#: test split vs the others' much lower rates). AUROC and PG2/PB2 are kept --
#: see the README's Severstal caveat.
_PREVALENCE_SENSITIVE_COLUMNS = ("mean_ap", "mean_f1")


def load_results(results_db: Path) -> pd.DataFrame:
    """Read the sweep's SQLite store into a frame.

    Raises:
        SystemExit: If the database is missing or holds no rows.
    """
    if not results_db.exists():
        msg = f"No results at {results_db}. Run training/benchmark/run.py first."
        raise SystemExit(msg)

    engine, _ = store.open_readonly(results_db)
    frame = pd.read_sql_table("runs", engine)
    if frame.empty:
        msg = f"{results_db} has no rows."
        raise SystemExit(msg)

    for column in (*_MEAN_COLUMNS, "peak_vram_gb"):
        if column not in frame:
            frame[column] = float("nan")
    return frame


def aggregate_seeds(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...] = ("method", "family", "backend", "config", "regime"),
) -> pd.DataFrame:
    """Collapse the seed dimension: every metric becomes ``{col}_mean``/``{col}_std``.

    ``std`` is ``NaN`` (not 0) for a cell with a single seed -- pandas' native
    ``ddof=1`` behavior needs no special-casing here, and a ``NaN`` std
    correctly reads as "no seed variability measured" rather than "measured
    and found to be zero".
    """
    metric_columns = list(dict.fromkeys([*_MEAN_COLUMNS, "peak_vram_gb"]))
    metric_columns = [c for c in metric_columns if c in frame.columns]
    grouped = frame.groupby(list(group_cols))
    agg = grouped[metric_columns].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_seeds"] = grouped.size()
    return agg.reset_index()


def rank_regime(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Build the per-method ranking for one regime.

    Two-level aggregation, per the spec: raw per-seed rows first collapse to
    one seed-mean per ``(method, config)`` cell (:func:`aggregate_seeds`),
    then this groupby averages *those cell means* across configs -- so a
    method that happened to get more successful seeds on one easy config
    cannot silently outweigh the rest.
    """
    subset = frame[(frame["status"] == "ok") & (frame["regime"] == regime)]
    if subset.empty:
        return pd.DataFrame()

    total_configs = frame[frame["regime"] == regime]["config"].nunique()

    cell_means = aggregate_seeds(subset, group_cols=("method", "family", "backend", "config"))

    aggregations: dict[str, Any] = {
        "configs": ("config", "nunique"),
        "min_auroc": ("auroc_mean", "min"),
        "peak_vram_gb": ("peak_vram_gb_mean", "max"),
    }
    aggregations.update({label: (f"{column}_mean", "mean") for column, label in _MEAN_COLUMNS.items()})
    # The "typical" std shown per method is the mean of its per-cell stds --
    # a summary of observed seed variability, not a properly pooled variance
    # (which would need per-cell sample sizes and isn't worth the complexity
    # for a leaderboard reading).
    aggregations.update(
        {f"{label}_std": (f"{column}_std", "mean") for column, label in _MEAN_COLUMNS.items()}
    )

    ranking = cell_means.groupby(["method", "family", "backend"]).agg(**aggregations).reset_index()
    ranking["complete"] = ranking["configs"] == total_configs
    # Complete runs rank above partial ones regardless of their mean, so a
    # method that only survived the easy categories cannot take the top slot.
    ranking = ranking.sort_values(["complete", "mean_auroc"], ascending=[False, False]).reset_index(
        drop=True
    )
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.insert(1, "regime", regime)

    # Severstal's held-out "normal" pool is synthesized (see datasets.py) and
    # its test split runs ~92% positive vs the other corpora's much lower
    # rate -- AP/F1 are prevalence-sensitive, so blending a Severstal row into
    # a multi-dataset mean_ap/mean_f1 would silently favor whichever dataset
    # mix a method happened to run on. AUROC and PG2/PB2 stay: less
    # prevalence-sensitive, and the whole reason those two exist. This only
    # fires when Severstal is actually mixed with other datasets in this
    # view -- a Severstal-only sweep keeps its AP/F1.
    datasets_in_view = set(subset["dataset"])
    if "severstal" in datasets_in_view and len(datasets_in_view) > 1:
        drop_columns = [
            column
            for label in _PREVALENCE_SENSITIVE_COLUMNS
            for column in (label, f"{label}_std")
            if column in ranking
        ]
        ranking = ranking.drop(columns=drop_columns)
        ranking.attrs["severstal_ap_f1_hidden"] = True
    return ranking


def shared_configs(frame: pd.DataFrame, regimes: list[str]) -> set[str]:
    """Configurations that every named regime actually covers.

    Multi-class is only defined for a family with more than one category, so
    it never covers the single-category corpora that one-class does. Comparing
    a 27-config mean against a 29-config mean would attribute the difference
    to the regime when part of it is just a different set of categories.
    """
    ok = frame[frame["status"] == "ok"]
    per_regime = [set(ok[ok["regime"] == regime]["config"]) for regime in regimes]
    per_regime = [configs for configs in per_regime if configs]
    if not per_regime:
        return set()
    return set.intersection(*per_regime)


def regime_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean AUROC per method per regime, over the configurations they share.

    This is the table the whole study exists to produce: a method whose
    multi-class column matches its one-class column is a method that ships as
    one checkpoint instead of one per category. It is restricted to shared
    configurations so the columns differ by regime alone. Seed-averaged first
    (one value per (method, config, regime) cell) so a multi-seed method isn't
    weighted by how many seeds happened to succeed.
    """
    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()

    regimes = sorted(set(ok["regime"]))
    common = shared_configs(frame, regimes)
    if not common:
        return pd.DataFrame()

    subset = ok[ok["config"].isin(common)]
    cell_means = aggregate_seeds(subset, group_cols=("method", "config", "regime"))
    table = (
        cell_means.pivot_table(index="method", columns="regime", values="auroc_mean", aggfunc="mean")
        .round(4)
        .reset_index()
    )
    sort_column = "multiclass" if "multiclass" in table.columns else table.columns[-1]
    return table.sort_values(by=sort_column, ascending=False)


def per_dataset(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Mean AUROC per method per dataset family, within one regime.

    Seed-averaged first, same reasoning as :func:`regime_comparison`.
    """
    subset = frame[(frame["status"] == "ok") & (frame["regime"] == regime)]
    if subset.empty:
        return pd.DataFrame()
    cell_means = aggregate_seeds(subset, group_cols=("method", "dataset", "regime"))
    return (
        cell_means.pivot_table(index="method", columns="dataset", values="auroc_mean", aggfunc="mean")
        .round(4)
        .reset_index()
    )


def winners(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Best method per configuration, within one regime.

    Ranked by each (method, config) cell's seed-averaged AUROC, so a single
    lucky seed can't make a mediocre method look like the winner.
    """
    subset = frame[(frame["status"] == "ok") & (frame["regime"] == regime)]
    if subset.empty:
        return pd.DataFrame()
    cell_means = aggregate_seeds(subset, group_cols=("method", "config", "regime"))
    best = cell_means.groupby("config")["auroc_mean"].idxmax()
    columns = [
        "config",
        "method",
        "auroc_mean",
        "average_precision_mean",
        "f1_max_mean",
        "ms_per_image_mean",
    ]
    picked = cell_means.loc[best, [c for c in columns if c in cell_means]].sort_values("config")
    return picked.rename(columns=lambda c: c.removesuffix("_mean"))


def _round(frame: pd.DataFrame, digits: dict[str, int]) -> pd.DataFrame:
    """Round selected columns for display, leaving missing ones alone."""
    display = frame.copy()
    for column, places in digits.items():
        if column in display:
            display[column] = display[column].round(places)
    return display


_DISPLAY_DIGITS = {
    "mean_auroc": 4,
    "min_auroc": 4,
    "mean_ap": 4,
    "mean_f1": 4,
    "scrap_at_95": 4,
    "escape_at_1fpr": 4,
    "mean_pg2": 4,
    "mean_pb2": 4,
    "bal_error": 4,
    "mean_aupro": 4,
    "mean_aupimo": 4,
    "mean_pixel_auroc": 4,
    "ms_per_image": 1,
    "fit_seconds": 1,
    "peak_vram_gb": 2,
    "latency_p50_ms": 1,
    "latency_p95_ms": 1,
    "throughput_fps_bs1": 1,
    "throughput_fps_bs16": 1,
    "model_params_m": 2,
    "peak_gpu_mb": 1,
    "train_wall_clock_s": 1,
}


def _with_mean_std_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace every ``{label}``/``{label}_std`` pair with one "mean +/- std" string.

    Display-only: the numeric ``_mean``/``_std`` columns stay untouched in
    whatever frame gets written to CSV -- this only ever runs on a copy meant
    for the Markdown report, per the spec's "mean +/- std, never a bare
    number" requirement.
    """
    display = frame.copy()
    for label in _MEAN_COLUMNS.values():
        std_label = f"{label}_std"
        if label not in display or std_label not in display:
            continue
        digits = _DISPLAY_DIGITS.get(label, 4)
        display[label] = [
            f"{mean:.{digits}f} +/- {std:.{digits}f}" if pd.notna(std) else f"{mean:.{digits}f}"
            for mean, std in zip(display[label], display[std_label], strict=True)
        ]
        display = display.drop(columns=[std_label])
    return display


def format_report(frame: pd.DataFrame, rankings: dict[str, pd.DataFrame]) -> str:
    """Render the summary tables as a Markdown report."""
    ok = frame[frame["status"] == "ok"]
    failed = frame[frame["status"] != "ok"]

    lines: list[str] = ["# Anomaly-detection model benchmark\n"]
    lines.append(
        f"{ok['method'].nunique()} methods across "
        f"{frame['config'].nunique()} dataset configurations and "
        f"{frame['regime'].nunique()} regimes -- "
        f"{len(ok)} successful runs, {len(failed)} failed.\n"
    )

    multiclass = rankings.get("multiclass", pd.DataFrame())
    oneclass = rankings.get("oneclass", pd.DataFrame())

    if not multiclass.empty:
        complete = multiclass[multiclass["complete"]]
        best = (complete if not complete.empty else multiclass).iloc[0]
        lines.append(
            f"**Headline -- one model, every category.** `{best['method']}` reaches "
            f"**{best['mean_auroc']:.4f} mean AUROC** across all {int(best['configs'])} "
            f"configurations from a single fitted model per dataset family "
            f"(worst config {best['min_auroc']:.4f}, {best['ms_per_image']:.1f} ms/image).\n"
        )
        if not oneclass.empty:
            # Compare on shared configurations only, or the gap conflates the
            # regime with the different set of categories each one covers.
            # Seed-averaged per cell first, same reasoning as rank_regime.
            common = shared_configs(frame, ["multiclass", "oneclass"])
            ok = frame[(frame["status"] == "ok") & (frame["config"].isin(common))]
            cell_means = aggregate_seeds(ok, group_cols=("method", "config", "regime"))
            multi_mean = cell_means[cell_means["regime"] == "multiclass"].groupby("method")[
                "auroc_mean"
            ].mean()
            one_mean = cell_means[cell_means["regime"] == "oneclass"].groupby("method")[
                "auroc_mean"
            ].mean()

            if not multi_mean.empty and not one_mean.empty:
                multi_best = multi_mean.idxmax()
                one_best = one_mean.idxmax()
                gap = multi_mean.max() - one_mean.max()
                lines.append(
                    f"On the {len(common)} configurations both regimes cover, the best "
                    f"multi-class model (`{multi_best}`, {multi_mean.max():.4f}) lands "
                    f"{gap * 100:+.2f} points against the best per-category model "
                    f"(`{one_best}`, {one_mean.max():.4f}) -- which needs one checkpoint "
                    f"per category rather than one per dataset family.\n"
                )

    comparison = regime_comparison(frame)
    if not comparison.empty:
        common = shared_configs(frame, sorted(set(frame[frame["status"] == "ok"]["regime"])))
        lines.append("## Regime comparison\n")
        lines.append(
            f"Mean AUROC per method under each deployment regime, over the "
            f"{len(common)} configurations every regime covers. A method whose "
            "multi-class column matches its one-class column ships as one artifact "
            "per dataset family instead of one per category.\n"
        )
        lines.append(comparison.to_markdown(index=False))
        lines.append("")

    for regime, ranking in rankings.items():
        if ranking.empty:
            continue
        lines.append(f"## Ranking -- {regime}\n")
        lines.append(
            "`scrap_at_95` is the share of good parts rejected when tuned to catch 95% of "
            "defects; `escape_at_1fpr` is the share of defects missed within a 1% "
            "false-alarm budget. Methods that did not complete every configuration are "
            "listed last and are not comparable to a complete row.\n"
        )
        if ranking.attrs.get("severstal_ap_f1_hidden"):
            lines.append(
                "`mean_ap`/`mean_f1` are omitted from this table: Severstal's test split runs "
                "~92% positive (see the Severstal caveat below), which would make a blended "
                "cross-dataset AP/F1 mean incomparable. AUROC and PG2/PB2 are less "
                "prevalence-sensitive and stay.\n"
            )
        lines.append(_with_mean_std_strings(_round(ranking, _DISPLAY_DIGITS)).to_markdown(index=False))
        lines.append("")

        table = per_dataset(frame, regime)
        if not table.empty:
            lines.append(f"### Mean AUROC by dataset family -- {regime}\n")
            lines.append(table.to_markdown(index=False))
            lines.append("")

        table = winners(frame, regime)
        if not table.empty:
            lines.append(f"### Best method per configuration -- {regime}\n")
            lines.append(
                _round(
                    table, {"auroc": 4, "average_precision": 4, "f1_max": 4, "ms_per_image": 1}
                ).to_markdown(index=False)
            )
            lines.append("")

    if not failed.empty:
        lines.append("## Failed runs\n")
        lines.append(
            "Recorded rather than dropped: a model that cannot complete a dataset has told "
            "you something about its deployability.\n"
        )
        summary = failed.groupby(["method", "config"])["error"].first().reset_index().head(60)
        lines.append(summary.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Parse arguments, summarize the sweep, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_db.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_results(args.results_db)
    regimes = [r for r in ("multiclass", "oneclass") if r in set(frame["regime"])]
    regimes += sorted(set(frame["regime"]) - set(regimes))
    rankings = {regime: rank_regime(frame, regime) for regime in regimes}

    if all(ranking.empty for ranking in rankings.values()):
        msg = "No successful runs to summarize."
        raise SystemExit(msg)

    report = format_report(frame, rankings)
    (output_dir / "leaderboard.md").write_text(report, encoding="utf-8")

    combined = pd.concat([r for r in rankings.values() if not r.empty], ignore_index=True)
    combined.to_csv(output_dir / "ranking.csv", index=False)
    regime_comparison(frame).to_csv(output_dir / "regime_comparison.csv", index=False)
    frame.to_csv(output_dir / "all_runs.csv", index=False)
    for regime in regimes:
        table = winners(frame, regime)
        if not table.empty:
            table.to_csv(output_dir / f"winners_{regime}.csv", index=False)

    print(report)
    print(f"\nWrote leaderboard.md, ranking.csv, regime_comparison.csv to {output_dir}")


if __name__ == "__main__":
    main()
