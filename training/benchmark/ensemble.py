"""Score fusion across methods, computed from stored artifacts.

Different families fail differently. A memory bank misses a defect whose
patches happen to resemble something in the normal set; a reconstruction model
misses one it has learned to redraw; a Gaussian misses one that stays inside
the normal ellipsoid. Where those failures are uncorrelated, combining the
scores recovers them -- which is why an ensemble is worth measuring even when
no member is best on its own.

This runs entirely on the ``.npz`` artifacts the sweep already wrote, so
evaluating every pair and triple costs seconds and no GPU. That matters: it
means the ensemble question is answered exhaustively rather than by guessing
which two methods to try.

Fusion happens on **ranks**, not raw scores. Members produce Mahalanobis
distances, nearest-neighbour distances and cosine errors, whose scales differ
by orders of magnitude -- averaging those directly would just return whichever
member has the largest numbers.

Usage:
    python training/benchmark/ensemble.py
    python training/benchmark/ensemble.py --regime multiclass --max-members 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # Allow `python training/benchmark/ensemble.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import store
from benchmark.artifacts import load_artifact
from benchmark.evaluation import compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"

#: Artifact filenames look like ``{method}__{config_slug}__seed{N}.npz``.
_ARTIFACT_NAME = re.compile(r"^(?P<method>.+)__(?P<slug>.+)__seed(?P<seed>\d+)$")

#: Candidate members are capped to the best few per family. Fusing two
#: backbones of the same method mostly averages correlated errors; the gain
#: comes from combining *different* families.
_PER_FAMILY_CANDIDATES = 2


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map scores to ``[0, 1]`` by rank, so members combine on equal footing.

    Ties receive their average rank, which keeps a member that emits many
    identical scores from silently dominating the fusion.
    """
    from scipy.stats import rankdata

    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    return (rankdata(scores) - 1) / (scores.size - 1)


def fuse(score_sets: list[np.ndarray], rule: str) -> np.ndarray:
    """Combine rank-normalized member scores under one fusion rule.

    Args:
        score_sets: One score array per member, all the same length.
        rule: ``"mean"``, ``"max"``, or ``"gmean"``.

    Returns:
        The fused score array.

    Raises:
        ValueError: If ``rule`` is unknown.
    """
    stacked = np.stack([rank_normalize(scores) for scores in score_sets])
    if rule == "mean":
        return stacked.mean(axis=0)
    if rule == "max":
        # Any member confidently flagging an image carries the decision.
        # Best recall, worst false-alarm rate.
        return stacked.max(axis=0)
    if rule == "gmean":
        # Geometric mean: members must broadly agree, so it suppresses the
        # lone-member false alarms that "max" lets through.
        return np.exp(np.log(np.clip(stacked, 1e-9, None)).mean(axis=0))
    msg = f"Unknown fusion rule: {rule!r}"
    raise ValueError(msg)


def discover_artifacts(artifact_root: Path, regime: str) -> dict[str, dict[str, Path]]:
    """Index stored artifacts as ``{config_key: {method: path}}``.

    Fusion stays single-seed: with the 3-seed repeats each producing its own
    ``__seed{N}.npz``, this keeps only the lowest seed per (config, method)
    rather than combinatorially fusing across seeds too, which would blow up
    the already-expensive combination search for a question ("does seed
    variance affect ensembling") this pass isn't trying to answer.
    """
    index: dict[str, dict[str, Path]] = defaultdict(dict)
    chosen_seed: dict[tuple[str, str], int] = {}
    regime_dir = artifact_root / regime
    if not regime_dir.is_dir():
        return index
    for path in sorted(regime_dir.glob("*.npz")):
        match = _ARTIFACT_NAME.match(path.stem)
        if match is None:
            continue
        method, slug, seed = match["method"], match["slug"], int(match["seed"])
        key = (slug, method)
        if key not in chosen_seed or seed < chosen_seed[key]:
            chosen_seed[key] = seed
            index[slug][method] = path
    return index


def choose_candidates(
    results_db: Path, regime: str, per_family: int = _PER_FAMILY_CANDIDATES
) -> list[str]:
    """Pick the strongest few methods per family as ensemble candidates.

    Args:
        results_db: The sweep's SQLite store.
        regime: Regime to read scores from.
        per_family: How many methods to keep from each family.

    Returns:
        Candidate method names.
    """
    from sqlalchemy import select as sa_select

    from benchmark.store import RunRow, session_scope

    _, session_factory = store.open_database(results_db)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    with session_scope(session_factory) as session:
        rows = session.scalars(
            sa_select(RunRow).where(RunRow.status == "ok", RunRow.regime == regime)
        )
        for row in rows:
            if row.auroc is None or row.auroc != row.auroc:  # missing or NaN
                continue
            scores[(row.family, row.method)].append(float(row.auroc))

    best: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (family, method), values in scores.items():
        best[family].append((float(np.mean(values)), method))

    candidates: list[str] = []
    for entries in best.values():
        entries.sort(reverse=True)
        candidates.extend(method for _, method in entries[:per_family])
    return sorted(candidates)


def evaluate_combinations(
    artifact_root: Path,
    regime: str,
    candidates: list[str],
    max_members: int,
    rules: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Score every candidate combination on every config it fully covers.

    A combination is only evaluated where *every* member has an artifact with
    matching labels; a partially-covered ensemble would be scored on an easier
    subset than its members were.

    Returns:
        One row per ``(combination, rule, config)``.
    """
    index = discover_artifacts(artifact_root, regime)
    rows: list[dict[str, Any]] = []

    combos: list[tuple[str, ...]] = []
    for size in range(2, max_members + 1):
        combos.extend(itertools.combinations(candidates, size))

    for slug, by_method in sorted(index.items()):
        # The membership guard has to precede the lookup: comprehension
        # conditions evaluate in written order, so testing it second would
        # raise KeyError on any candidate this config never ran.
        loaded = {
            method: artifact
            for method in candidates
            if method in by_method
            if (artifact := load_artifact(by_method[method])) is not None
        }
        if len(loaded) < 2:
            continue

        for combo in combos:
            if not all(method in loaded for method in combo):
                continue
            members = [loaded[method] for method in combo]
            labels = members[0].labels
            if any(
                m.labels.shape != labels.shape or not np.array_equal(m.labels, labels)
                for m in members
            ):
                continue

            for rule in rules:
                fused = fuse([m.scores for m in members], rule)
                metrics = compute_metrics(fused, labels)
                rows.append(
                    {
                        "regime": regime,
                        "method": f"ensemble[{rule}]:" + "+".join(combo),
                        "family": "ensemble",
                        "backend": "ensemble",
                        "config": slug.replace("_", "/", 1) if "_" in slug else slug,
                        "members": list(combo),
                        "rule": rule,
                        "n_members": len(combo),
                        "status": "ok",
                        **metrics.as_dict(),
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-config ensemble rows into a per-combination ranking."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary = [
        {
            "method": method,
            "rule": entries[0]["rule"],
            "n_members": entries[0]["n_members"],
            "members": entries[0]["members"],
            "configs": len(entries),
            "mean_auroc": float(np.mean([e["auroc"] for e in entries])),
            "mean_ap": float(np.mean([e["average_precision"] for e in entries])),
            "mean_f1": float(np.mean([e["f1_max"] for e in entries])),
            "mean_scrap_at_95": float(np.mean([e["fpr_at_95tpr"] for e in entries])),
        }
        for method, entries in grouped.items()
    ]
    summary.sort(key=lambda row: (-row["configs"], -row["mean_auroc"]))
    return summary


def main() -> None:
    """Parse arguments, evaluate ensembles, and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--artifacts", type=Path, default=RESULTS_DIR / "artifacts")
    parser.add_argument("--regime", default="oneclass")
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--rules", nargs="+", default=["mean", "max", "gmean"])
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ensembles.jsonl")
    args = parser.parse_args()

    if not args.results_db.exists():
        msg = f"No results at {args.results_db}. Run training/benchmark/run.py first."
        raise SystemExit(msg)

    candidates = choose_candidates(args.results_db, args.regime)
    if len(candidates) < 2:
        msg = f"Need at least two successful methods in regime {args.regime!r} to ensemble."
        raise SystemExit(msg)

    print(f"candidates ({len(candidates)}): {', '.join(candidates)}")
    rows = evaluate_combinations(
        args.artifacts, args.regime, candidates, args.max_members, tuple(args.rules)
    )
    if not rows:
        msg = "No artifacts covered a full combination. Was the sweep run with artifacts enabled?"
        raise SystemExit(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")

    summary = summarize(rows)
    print(f"\n{len(summary)} combinations over {len({r['config'] for r in rows})} configs\n")
    print(f"{'combination':<64s} {'cfgs':>5s} {'AUROC':>8s} {'F1':>8s} {'scrap@95':>9s}")
    for row in summary[:20]:
        print(
            f"{row['method']:<64.64s} {row['configs']:>5d} "
            f"{row['mean_auroc']:>8.4f} {row['mean_f1']:>8.4f} {row['mean_scrap_at_95']:>9.4f}"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
