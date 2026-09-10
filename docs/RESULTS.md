# Results

Numbers here come from the committed benchmark summary (`docs/benchmarks/`) and the 29 exported-model
manifests (`models/*.json`). The raw sweep database and prediction artifacts are not in the
repository; they are large and rebuildable.

## Benchmark setup

- **Methods:** PatchCore, PaDiM, DFM (implemented in `training/models.py`), Dinomaly (native,
  research-only), and optional Anomalib delegations. Each family is run on several `timm` backbones.
- **Datasets:** MVTec AD (15 categories), VisA (12 objects), KolektorSDD2, Severstal. 29 dataset
  configurations in total. Datasets are not included; see the README for sources.
- **Regimes:**
  - *one-class* — one fitted model per category (the setting most papers report).
  - *multi-class* — one fitted model per dataset family, scored on every category. Fewer models to
    version and deploy.
- **Seeds:** three per configuration.
- **Metrics** (`training/evaluate.py`, own implementations): image-level AUROC, average precision,
  F1-max; pixel-level AUROC, AUPRO, AUPIMO where masks exist; and two operating-point numbers —
  `scrap_at_95` (good parts rejected when catching 95% of defects) and `escape_at_1fpr` (defects
  missed within a 1% false-alarm budget).

## Sweep scale

From `docs/benchmarks/leaderboard.md` and the results database:

| | Count |
|---|---|
| Successful runs | 3,428 |
| Failed runs | 655 |
| Runs left in a crashed "running" state | 15 |

Every one of the 3,428 successful runs is a **native** method (implemented in this repo). Of the 655
failures, **648 are the optional Anomalib backend** — most of them the same error
(`MisconfigurationException: Trainer was configured with enable_checkpointing=False but found
ModelCheckpoint in callbacks`), a version-compatibility problem in that integration rather than in
the methods themselves. The other 7 failures are DFM runs that ran out of memory on the largest
split. The failures are recorded rather than dropped.

## Selected results

### Multi-class (one model per dataset family)

From `docs/benchmarks/leaderboard.md`, mean image AUROC over the 27 MVTec + VisA configurations
covered by both regimes:

| Method | Multi-class mean AUROC |
|---|---|
| Dinomaly (DINOv2-ViT-B/14) | 0.977 |
| Dinomaly (DINOv2-ViT-S/14) | 0.968 |
| PatchCore (dense, WideResNet50-2) | 0.894 |
| PatchCore (DINOv2-ViT-B/14) | 0.888 |
| DFM (DINOv2-ViT-B/14) | 0.840 |
| PaDiM (WideResNet50-2) | 0.737 |

Dinomaly's worst single configuration is 0.891 and it runs at about 16 ms/image (GPU). It is
implemented natively here but is not exported to ONNX and is not used by the runtime application —
it needs gradient training and a GPU.

### One-class (one model per category)

Best complete method by mean AUROC over all 29 configurations:

| Method | One-class mean AUROC |
|---|---|
| DFM (DINOv2-ViT-B/14) | 0.933 |
| PatchCore (DINOv2-ViT-L/14) | 0.924 |
| PatchCore (DINOv2-ViT-B/14) | 0.924 |
| PaDiM (WideResNet50-2) | 0.838 |

The per-category winner varies; see the "Best method per configuration" tables in
`docs/benchmarks/leaderboard.md`.

## Exported models

29 configurations were exported to ONNX with score calibration. Image-level AUROC recorded in each
manifest (`models/*.json`), from the sweep run the export was based on:

| Statistic across the 29 exports | Value |
|---|---|
| Mean AUROC | 0.910 |
| Min | 0.601 (PatchCore/DINOv2-B on `visa/macaroni2`) |
| Max | 1.000 (PatchCore/DINOv2-B on `mvtec/grid`) |

A few representative rows:

| Model | Config | AUROC |
|---|---|---|
| PatchCore / DINOv2-ViT-B/14 | mvtec/bottle | 0.9997 |
| PatchCore / DINOv2-ViT-B/14 | mvtec/hazelnut | 0.9998 |
| PatchCore / DINOv2-ViT-L/14 | mvtec/carpet | 0.9985 |
| DFM / WideResNet50-2 | visa/cashew | 0.9794 |
| PaDiM / WideResNet50-2 | kolektor | 0.9544 |
| PatchCore / DINOv2-ViT-B/14 | mvtec/screw | 0.7605 |
| DFM / WideResNet50-2 | severstal | 0.7291 |
| PatchCore / DINOv2-ViT-B/14 | visa/macaroni2 | 0.6008 |

MVTec categories are close to saturated for these methods; VisA (especially the macaroni objects)
and Severstal are where they are weak. `screw` is a known-hard MVTec category.

The `bottle` model is the one the demo (`scripts/run_station.py`) and the real-model e2e tests use.

## Quantization

`docs/benchmarks/quantization_comparison.md` (dynamic weights-only INT8, MatMul layers only, CPU
inference). The headline: naively quantizing an already-fitted PatchCore drops AUROC to ~0.5,
because the fp32 memory bank is compared against INT8 queries. Rebuilding the bank in the quantized
model's feature space recovers accuracy — the recalibrated INT8 models match or slightly exceed fp32
on the tested configurations, at roughly 73% smaller size and lower CPU p50 latency. This is a
measured comparison on a handful of configurations, not the full sweep.

## Runtime verification (2026-09-10, dev machine, CPU)

- `python scripts/run_station.py` on a good MVTec bottle image → verdict PASS.
- Same on `broken_large/000.png` → verdict FAIL, with a background LLM advisory report stored
  (`is_fallback: false`, `qwen2.5:7b`).
- Teach-mode upload of `broken_small/000.png` → verdict FAIL, anomaly score ~1.0, heatmap region
  `center-left`, one measured defect region (bbox, area, aspect ratio, morphology `particle`),
  original frame and heatmap archived as PNG.
- `pytest` → 528 passed, 95.42% branch coverage. `mypy` and `ruff check` clean.
