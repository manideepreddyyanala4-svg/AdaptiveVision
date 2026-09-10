# Benchmark summaries

Human-readable output from the anomaly-detection sweep (`training/sweep.py` +
`training/evaluate.py`). The raw results database, per-run prediction artifacts, and model
checkpoints are not committed — they are large and can be regenerated.

| File | Contents |
|---|---|
| `leaderboard.md` | the full report: regime comparison, per-method ranking, per-dataset and per-category breakdowns, and the list of failed runs |
| `ranking.csv` | the aggregated ranking table (all metrics, both regimes) |
| `regime_comparison.csv` | mean AUROC per method under one-class vs multi-class |
| `winners_oneclass.csv`, `winners_multiclass.csv` | best method per dataset configuration in each regime |
| `winners_exportable.csv` | best *ONNX-exportable* method per configuration (drops Dinomaly and Anomalib) — this is what `training/export.py --from-leaderboard` reads |
| `quantization_comparison.md` | fp32 vs recalibrated INT8 on a subset of configurations |

See `../RESULTS.md` for a summary and `../ARCHITECTURE.md` for how the sweep fits together.
