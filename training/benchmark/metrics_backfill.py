"""Backfill PG2/PB2/AUPIMO onto existing rows -- no rerun required.

Both new metrics can be computed from what the sweep already saved:

* PG2/PB2 (image-level) need only the raw scores/labels already in the
  ``.npz`` artifact.
* AUPIMO (pixel-level) needs the artifact's saved anomaly maps paired with
  freshly-loaded ground-truth masks -- the ``.npz`` doesn't store masks
  (only scores/labels/paths/maps), so this reloads them from disk the same
  way the sweep did (cheap: no model, no GPU, just image I/O), using the
  same class-balanced sampling as the sweep's own pixel-metrics pass so the
  backfilled number reflects the same sample the original aupro was scored
  against.

Run as a stage after the sweep:

    python training/benchmark/metrics_backfill.py --results-db training/benchmark_results/benchmark.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # Allow `python training/benchmark/metrics_backfill.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import store
from benchmark.artifacts import artifact_path, load_artifact
from benchmark.data import DatasetConfig
from benchmark.evaluation import compute_metrics, compute_pixel_metrics
from benchmark.masks import has_masks, load_mask
from benchmark.regimes import _pixel_sample_indices  # reuse the sweep's own sampling

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"
DEFAULT_ARTIFACTS = RESULTS_DIR / "artifacts"


def _pixel_backfill(artifact, config: DatasetConfig, data_root: Path, seed: int) -> dict[str, float] | None:
    """Recompute pixel metrics (for aupimo) over the same sample the sweep used."""
    if artifact.maps is None or not has_masks(config):
        return None
    test_split = [(Path(p), bool(label)) for p, label in zip(artifact.paths, artifact.labels)]
    indices = _pixel_sample_indices(test_split, seed)

    selected_maps = []
    selected_masks = []
    for index in indices:
        mask = load_mask(test_split[index][0], config, data_root, config.height, config.width)
        if mask is None:
            continue
        selected_maps.append(artifact.maps[index])
        selected_masks.append(mask)

    if not selected_masks:
        return None

    import numpy as np

    pixel = compute_pixel_metrics(np.stack(selected_maps), np.stack(selected_masks))
    return {"aupimo": pixel.aupimo}


def run_metrics_pass(results_db: Path, artifact_root: Path, data_root: Path) -> tuple[int, int]:
    """Visit every completed row missing pg2 and backfill pg2/pb2 (+ aupimo where masks exist).

    Returns:
        ``(updated, skipped_no_artifact)`` counts.
    """
    from sqlalchemy import select as sa_select

    from benchmark.store import RunRow, session_scope

    _, session_factory = store.open_database(results_db)

    with session_scope(session_factory) as session:
        rows = list(
            session.scalars(sa_select(RunRow).where(RunRow.status == "ok", RunRow.pg2.is_(None)))
        )
        pending = [
            (
                row.run_id,
                row.regime,
                row.method,
                row.config,
                row.dataset,
                row.category,
                row.height,
                row.width,
                row.seed,
            )
            for row in rows
        ]

    updated = 0
    skipped = 0
    for index, (run_id, regime, method, config_key, dataset, category, height, width, seed) in enumerate(
        pending, start=1
    ):
        path = artifact_path(artifact_root, regime, method, config_key, seed)
        artifact = load_artifact(path)
        if artifact is None:
            print(f"[{index}/{len(pending)}] {method} @ {config_key}: no artifact at {path}, skipping")
            skipped += 1
            continue

        columns: dict[str, float] = {}
        if artifact.labels.any() and not artifact.labels.all():
            image_metrics = compute_metrics(artifact.scores, artifact.labels)
            columns["pg2"] = image_metrics.pg2
            columns["pb2"] = image_metrics.pb2

        config = DatasetConfig(
            dataset=dataset, category=category, height=height, width=width, position_aligned=True
        )
        pixel_columns = _pixel_backfill(artifact, config, data_root, seed)
        if pixel_columns:
            columns.update(pixel_columns)

        if columns:
            store.update_columns(session_factory, run_id, columns)
            print(f"[{index}/{len(pending)}] {method} @ {config_key}: {columns}")
            updated += 1
        else:
            skipped += 1

    return updated, skipped


def main() -> None:
    """Parse arguments and run the metrics backfill pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    args = parser.parse_args()

    updated, skipped = run_metrics_pass(args.results_db, args.artifacts, args.data_root)
    print(f"\nbackfilled pg2/pb2(/aupimo) on {updated} row(s), {skipped} skipped")


if __name__ == "__main__":
    main()
