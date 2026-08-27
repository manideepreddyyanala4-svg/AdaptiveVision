"""Live view of the benchmark sweep, reading directly from its SQLite store.

Safe to run while ``run_all.py`` is still writing to the same database (a
read-only ``mode=ro`` connection onto the writer's WAL-mode file -- see
``benchmark.store.open_readonly``). Auto-refreshes every 30 seconds.

Usage:
    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --results-db path/to/benchmark.db
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = str(REPO_ROOT / "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from benchmark import store  # noqa: E402
from benchmark.leaderboard import _MEAN_COLUMNS, aggregate_seeds  # noqa: E402

DEFAULT_RESULTS_DB = REPO_ROOT / "training" / "benchmark_results" / "benchmark.db"

#: Prevalence-sensitive metrics -- hidden from any view spanning multiple
#: dataset families, matching leaderboard.py's rank_regime restriction (§5).
_PREVALENCE_SENSITIVE = ("mean_ap", "mean_f1")

st.set_page_config(page_title="Anomaly Detection Benchmark", layout="wide")


def _results_db_path() -> Path:
    """``--results-db`` from the CLI passed after ``--`` to ``streamlit run``, or the default."""
    args = sys.argv[1:]
    if "--results-db" in args:
        return Path(args[args.index("--results-db") + 1])
    return DEFAULT_RESULTS_DB


@st.cache_data(ttl=30)
def load_data(db_path: str) -> pd.DataFrame:
    """Read every completed row, cached for the same 30s window as the auto-refresh."""
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    engine, _ = store.open_readonly(path)
    frame = pd.read_sql_table("runs", engine)
    return frame[frame["status"] == "ok"] if not frame.empty else frame


@st.cache_data(ttl=30)
def load_cell_means(db_path: str) -> pd.DataFrame:
    """Seed-collapsed means -- one row per (method, family, backend, config, dataset, regime)."""
    frame = load_data(db_path)
    if frame.empty:
        return frame
    return aggregate_seeds(
        frame, group_cols=("method", "family", "backend", "config", "dataset", "regime")
    )


def _hide_prevalence_sensitive(columns: list[str], datasets_in_view: set[str]) -> list[str]:
    """Drop AP/F1 columns from a view spanning >1 dataset family, incl. Severstal."""
    if "severstal" in datasets_in_view and len(datasets_in_view) > 1:
        blocked = {f"{label}_mean" for label in _PREVALENCE_SENSITIVE} | {
            f"{label}_std" for label in _PREVALENCE_SENSITIVE
        }
        return [c for c in columns if c not in blocked]
    return columns


def section_deployment_recommender(cell_means: pd.DataFrame) -> None:
    """(a) The headline feature: rank configs against a deployment budget."""
    st.header("Deployment Recommender")
    if cell_means.empty:
        st.info("No completed runs yet.")
        return

    datasets = sorted(cell_means["dataset"].unique())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dataset = st.selectbox("Dataset", ["(any)", *datasets])
    with col2:
        max_latency = st.number_input("Max latency p50 (ms)", min_value=0.0, value=50.0, step=1.0)
    with col3:
        min_auroc = st.slider("Min AUROC", 0.0, 1.0, 0.9, 0.01)
    with col4:
        max_size_mb = st.number_input("Max model size (MB)", min_value=0.0, value=500.0, step=10.0)

    view = cell_means
    if dataset != "(any)":
        view = view[view["dataset"] == dataset]
    view = view.dropna(subset=["inference_latency_ms_p50_mean", "auroc_mean"])
    matches = view[
        (view["inference_latency_ms_p50_mean"] <= max_latency)
        & (view["auroc_mean"] >= min_auroc)
        & (view.get("model_params_millions_mean", pd.Series(dtype=float)).fillna(0) * 4 <= max_size_mb)
    ].sort_values("inference_latency_ms_p50_mean")

    if matches.empty:
        st.warning("No config satisfies every constraint. Try relaxing one.")
        return

    def _explain(row: pd.Series, rank: int) -> str:
        if rank == 0:
            return "fastest option above your accuracy floor"
        if row["auroc_mean"] == matches["auroc_mean"].max():
            return "best accuracy within your latency budget"
        return "meets every constraint"

    display = matches.head(15).copy()
    display["why"] = [
        _explain(row, i) for i, (_, row) in enumerate(display.iterrows())
    ]
    display["accuracy"] = display.apply(
        lambda r: f"{r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}"
        if pd.notna(r.get("auroc_std"))
        else f"{r['auroc_mean']:.4f}",
        axis=1,
    )
    st.dataframe(
        display[
            [
                "method",
                "family",
                "config",
                "accuracy",
                "inference_latency_ms_p50_mean",
                "inference_latency_ms_p95_mean",
                "model_params_millions_mean",
                "why",
            ]
        ].rename(
            columns={
                "inference_latency_ms_p50_mean": "latency p50 (ms)",
                "inference_latency_ms_p95_mean": "latency p95 (ms)",
                "model_params_millions_mean": "params (M)",
            }
        ),
        hide_index=True,
        width='stretch',
    )


def section_pareto_frontier(cell_means: pd.DataFrame) -> None:
    """(b) Accuracy vs. latency, with the non-dominated frontier highlighted."""
    st.header("Accuracy vs. Latency Pareto Frontier")
    if cell_means.empty:
        st.info("No completed runs yet.")
        return

    datasets = sorted(cell_means["dataset"].unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        chosen_datasets = st.multiselect("Datasets", datasets, default=datasets)
    with col2:
        regimes = sorted(cell_means["regime"].unique())
        regime = st.radio("Regime", regimes, horizontal=True)
    with col3:
        metric_options = ["auroc_mean"]
        if not _hide_prevalence_sensitive(["mean_ap"], set(chosen_datasets)) == []:
            metric_options += ["average_precision_mean", "f1_max_mean"]
        metric = st.selectbox("Accuracy metric", metric_options)

    view = cell_means[
        cell_means["dataset"].isin(chosen_datasets) & (cell_means["regime"] == regime)
    ].dropna(subset=["inference_latency_ms_p50_mean", metric])
    if view.empty:
        st.info("No rows match this filter.")
        return

    view = view.sort_values("inference_latency_ms_p50_mean")
    frontier_mask = view[metric].cummax() == view[metric]
    view = view.assign(frontier=frontier_mask)

    fig = px.scatter(
        view,
        x="inference_latency_ms_p50_mean",
        y=metric,
        size=view["model_params_millions_mean"].fillna(1.0).clip(lower=0.1),
        color="family",
        symbol=view["frontier"].map({True: "star", False: "circle"}),
        hover_data=["method", "config"],
        log_x=True,
        labels={"inference_latency_ms_p50_mean": "latency p50 (ms, log scale)", metric: metric},
    )
    frontier_points = view[view["frontier"]].sort_values("inference_latency_ms_p50_mean")
    fig.add_scatter(
        x=frontier_points["inference_latency_ms_p50_mean"],
        y=frontier_points[metric],
        mode="lines",
        name="Pareto frontier",
        line={"dash": "dot"},
    )
    st.plotly_chart(fig, width='stretch')


def section_leaderboard_table(cell_means: pd.DataFrame) -> None:
    """(c) Full sortable/filterable table, mean +/- std, CSV export."""
    st.header("Leaderboard")
    if cell_means.empty:
        st.info("No completed runs yet.")
        return

    datasets = sorted(cell_means["dataset"].unique())
    col1, col2 = st.columns(2)
    with col1:
        chosen_datasets = st.multiselect(
            "Datasets", datasets, default=datasets, key="leaderboard_datasets"
        )
    with col2:
        regimes = sorted(cell_means["regime"].unique())
        chosen_regimes = st.multiselect("Regimes", regimes, default=regimes)

    view = cell_means[
        cell_means["dataset"].isin(chosen_datasets) & cell_means["regime"].isin(chosen_regimes)
    ]
    mean_cols = [c for c in view.columns if c.endswith("_mean")]
    mean_cols = _hide_prevalence_sensitive(mean_cols, set(chosen_datasets))
    id_cols = ["method", "family", "backend", "dataset", "config", "regime", "n_seeds"]
    st.dataframe(view[id_cols + mean_cols], hide_index=True, width='stretch')
    st.download_button(
        "Download CSV", view.to_csv(index=False), file_name="leaderboard.csv", mime="text/csv"
    )


def section_fewshot_curves(raw: pd.DataFrame) -> None:
    """(d) Accuracy vs. shot count, one line per method, faceted by dataset."""
    st.header("Few-Shot Curves")
    fewshot = raw[raw["regime"].str.startswith("fewshot", na=False)]
    if fewshot.empty:
        st.info("No few-shot runs yet.")
        return
    fig = px.line(
        fewshot.sort_values("n_shot"),
        x="n_shot",
        y="auroc",
        color="method",
        facet_col="dataset",
        markers=True,
        log_x=True,
        labels={"n_shot": "shots (k, log scale)", "auroc": "AUROC"},
    )
    st.plotly_chart(fig, width='stretch')


def section_drift_stub() -> None:
    """(e) Stubbed per decision -- no drift/augmentation experiment exists in this codebase."""
    st.header("Robustness / Drift Comparison")
    st.info(
        "Not implemented: no drift/augmentation experiment exists in this codebase yet. "
        "See the README's caveats section -- future work."
    )


def section_dataset_difficulty_note(cell_means: pd.DataFrame) -> None:
    """(f) A stated finding, not buried in a table: MVTec/VisA are largely saturated."""
    st.header("Dataset Difficulty")
    st.markdown(
        "MVTec AD and VisA are largely saturated by 2025-era methods -- see "
        "Bertoldo et al., BMVC 2024, and Heckler-Kram et al., 2025. A method that "
        "looks state-of-the-art on either benchmark alone may not actually be "
        "state-of-the-art; MVTec LOCO (harder, structural/logical anomalies) is "
        "the more informative comparison once it's in the sweep."
    )
    if cell_means.empty:
        return
    families_present = set(cell_means["dataset"].unique())
    saturated = {"mvtec", "visa"} & families_present
    harder = {"mvtec_loco"} & families_present
    if saturated:
        saturated_mean = cell_means[cell_means["dataset"].isin(saturated)]["auroc_mean"].mean()
        st.metric("Mean AUROC -- MVTec/VisA (saturated)", f"{saturated_mean:.4f}")
    if harder:
        harder_mean = cell_means[cell_means["dataset"].isin(harder)]["auroc_mean"].mean()
        st.metric("Mean AUROC -- MVTec LOCO (harder)", f"{harder_mean:.4f}")
    elif saturated:
        st.caption("MVTec LOCO not in the sweep yet -- no harder comparison to show.")


def main() -> None:
    """Render every section in the spec's order, auto-refreshing every 30s."""
    st_autorefresh(interval=30_000, key="autorefresh")
    st.title("Anomaly Detection Benchmark -- Live")

    db_path = _results_db_path()
    st.caption(f"Reading {db_path} (read-only, refreshes every 30s)")

    raw = load_data(str(db_path))
    cell_means = load_cell_means(str(db_path))

    section_deployment_recommender(cell_means)
    section_pareto_frontier(cell_means)
    section_leaderboard_table(cell_means)
    section_fewshot_curves(raw)
    section_drift_stub()
    section_dataset_difficulty_note(cell_means)


if __name__ == "__main__":
    main()
