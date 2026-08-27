# Industrial anomaly detection — full study

Benchmarks a zoo of anomaly-detection methods across **every dataset configuration
on disk**, in **three deployment regimes**, and produces a ranked leaderboard, a live
dashboard, and a deployable ONNX model.

```bash
pip install -r training/requirements.txt
python training/benchmark/run_all.py
```

That one command runs the whole pipeline. Launch the live dashboard separately (it's
a server, not a batch step) to watch progress while it runs:

```bash
streamlit run dashboard/app.py
```

---

## What it covers

**29 dataset configurations on disk today** (plus MVTec LOCO once downloaded — see
below)

| Corpus | Configurations |
| --- | --- |
| MVTec AD | 15 categories (`bottle` … `zipper`) |
| VisA | 12 objects (`candle` … `pipe_fryum`) |
| KolektorSDD2 | 1 |
| Severstal steel | 1 |
| MVTec LOCO | 5 objects, once downloaded — see [Caveats](#caveats) |

**14 native methods**, trimmed from an earlier, larger sweep to cut compute waste
(see below), plus ~26 more if Anomalib is installed.

| Family | Scoring rule | Variants |
| --- | --- | --- |
| `dinomaly` | Frozen DINOv2 + linear-attention decoder; anomalies are where reconstruction fails | ViT-S/B/L, **multi-class only** |
| `patchcore` | Nearest-neighbour distance to a coreset memory bank | ResNet18, WideResNet50-2, ConvNeXt-S, DINOv2-ViT-B/14 + dense-coreset/3-NN variants on WideResNet50-2 |
| `padim` | Mahalanobis distance to a per-position Gaussian | WideResNet50-2 + 2 pooled (position-agnostic) variants |
| `dfm` | PCA reconstruction error on pooled features (cheap control) | 2 backbones, unchanged — the most valuable training-free baseline in the sweep |

**Why Dinomaly is multi-class only.** Guo et al., "Dinomaly: The Less Is More
Philosophy in Multi-Class Unsupervised Anomaly Detection," CVPR 2025, Table 2, report
a <0.2pp gap between one-class and multi-class Dinomaly on MVTec/VisA — within the
paper's own 5-seed noise band (±0.03). Running one-class here would just spend GPU
hours re-measuring noise, so it's cut from the sweep entirely (see
`methods_dinomaly.py`'s registration comment).

**Three regimes, plus a light few-shot pass**:

- **one-class** — one model per category. What almost every paper reports, and the
  easiest to win: each model only has to represent one product. Also N checkpoints to
  version, deploy, calibrate and monitor.
- **multi-class** — *one* model per dataset family, covering every category with no
  category label at fit or inference time. One checkpoint, one deployment.
- **few-shot** — fit on k ∈ {1,2,4,8,16} normal images. The cold-start case: a new
  product arrives and nobody has collected a thousand good samples yet. Restricted to
  **training-free methods on MVTec/VisA only** (PatchCore/PaDiM/DFM) — a
  gradient-trained model on 1-16 images isn't a meaningful result, and the question
  this pass answers is about product variety, not defect rarity. Single seed, not 3 —
  see [Caveats](#caveats).

**Every kept combination runs 3 seeds** (1, 2, 3); every table and the dashboard show
**mean ± std**, never a bare number.

---

## Run it

```bash
python training/benchmark/run_all.py --quick
```

Smoke test first — 3 methods, 2 categories, 1 seed, a few minutes. It exercises the
whole pipeline including the parts that only break on real data (mask loading,
artifact/checkpoint round-trips). Then the real thing:

```bash
python training/benchmark/run_all.py
```

Stages, all individually skippable with `--skip`:

| Stage | What it does | Cost |
| --- | --- | --- |
| `sweep` | Fits and scores every method × config × regime × seed | hours-days |
| `cost` | Loads each run's saved checkpoint and times its real forward pass (latency/throughput/params/VRAM), backfilled onto the same row | minutes |
| `metrics` | Backfills PG2/PB2/AUPIMO from stored `.npz` artifacts, for any row that predates those metrics | seconds |
| `leaderboard` | Ranks it, writes `leaderboard.md` | seconds |
| `ensemble` | Fuses methods from stored artifacts | seconds, no GPU |
| `deploy` | Separate multi-device (CPU **and** GPU) latency/VRAM/size comparison for the top methods | minutes |

The static `dashboard.py` HTML builder from earlier milestones still exists but is
superseded by the live Streamlit app (`dashboard/app.py`) — it isn't a `run_all.py`
stage any more.

### Resumability

The sweep is **crash-safe and resumable at the individual-run level**, not just
per-stage. Every result row is keyed by a deterministic `run_id` (a hash of method +
regime + config + seed + a couple of dataset-specific fields); before a run starts,
its row is inserted as `status="running"`, then updated to `"ok"`/`"failed"` the
moment it finishes, committed immediately — no batching, so a kill loses at most the
one in-flight run. Relaunching the identical command:

```bash
python training/benchmark/run_all.py
```

re-reads the SQLite store (`training/benchmark_results/benchmark.db`), deletes any
row still `"running"` or `"failed"` (a crash victim — indistinguishable from
never-attempted, so it's always retried), and continues from exactly where it left
off. **This is true after a reboot too** — nothing about resuming depends on the
process or the machine having stayed up. Safe to `Ctrl+C` and rerun any time,
including for whoever restarts this job next.

Two flags for inspecting or narrowing the plan without running anything:

```bash
python training/benchmark/run_all.py --dry-run                                   # print the plan + ETA, run nothing
python training/benchmark/run_all.py --only method=patchcore dataset=mvtec_loco  # targeted reruns
```

Useful flags: `--models`, `--datasets`, `--regimes`, `--max-fit-images`,
`--batch-size`, `--epochs`, `--no-pixel` (skips localization, noticeably faster),
`--force`, `--seeds` (nargs, default `1 2 3` — **note: this replaced the old singular
`--seed`**), `--severstal-target-prevalence`.

### Runtime on a 16 GB card

TODO: refresh once the first full 3-seed run (incl. few-shot) completes — this trim +
seed-repeat + few-shot combination hasn't finished end-to-end yet. Notes so far:

- Dinomaly trains for 10k iterations per fit; that is the bulk of it. `--epochs 3000`
  cuts it roughly threefold at a small accuracy cost.
- Severstal's test split is ~7,250 images. `--max-test-images 2000` (the default in
  `run_all.py`) saves hours and barely moves AUROC.
- 3 seeds × the trimmed roster is cheaper than the old 21-method single-seed sweep for
  the training-free methods, but Dinomaly's multi-class-only training now runs 3×.
- Adding Anomalib's gradient-trained zoo at full epochs across every config is days,
  not hours. Stage it: `--models anomalib --datasets mvtec --epochs 20`.

---

## Outputs

Everything lands in `training/benchmark_results/`:

| File | Contents |
| --- | --- |
| `benchmark.db` | **The source of truth.** SQLite, one row per (method, regime, config, seed) — see Resumability |
| `leaderboard.md` | Written report with every table, mean ± std |
| `ranking.csv` | Per-regime ranking |
| `regime_comparison.csv` | One-class vs multi-class vs few-shot, like-for-like |
| `artifacts/` | Raw scores, labels and heatmaps per run, one `.npz` per (regime, method, config, **seed**) |
| `checkpoints/` | The fitted model itself, one `.pt` per (regime, method, config) fit — see below |

`artifacts/` is what makes the sweep's *accuracy* pass a one-time cost: the
dashboard's ROC curves, every ensemble combination, and PG2/PB2/AUPIMO are all
computed from it without touching a GPU. `checkpoints/` is what makes the *cost* pass
a one-time cost too: `cost.py` loads the saved model and times its real forward pass
instead of re-fitting just to get something to time. Multi-class checkpoints are
saved once per family (every category's row shares that one fit), and few-shot
checkpoints are keyed by shot count as well.

---

## Metrics

AUROC ranks models. It does not tell you what a line costs, so the report leads with
the two numbers that do:

- **Scrap@95** — share of good parts rejected when tuned to catch 95% of defects.
- **Escape@1** — share of defects missed within a 1% false-alarm budget.
- **PG2 / PB2** (Baitieva et al., "Beyond Academic Benchmarks," 2025) — a second,
  2%-budget operating point: PG2 is the share of good parts correctly passed when the
  threshold is set to catch all but 2% of bad parts; PB2 is the mirror, the share of
  bad parts correctly rejected when only 2% of good parts may false-alarm. A
  different operating point from Scrap@95/Escape@1, not a replacement.

Plus AP, F1-max with its threshold, balanced error rate, and for localization
**pixel AUROC, AUPRO, and AUPIMO**. AUPRO matters because pixel AUROC is dominated by
large defects — a big region simply contributes more pixels — while AUPRO weights
every defect region equally. AUPIMO (Bertoldo et al., BMVC 2024) goes further: AUPRO
still pools every region and every normal pixel across the *whole batch* into one FPR
curve, so one image's abundant normal pixels can dominate the curve every other
image's regions get scored against; AUPIMO integrates each image's regions against
that same image's own normal-pixel range, then averages the per-image scores. Vendored
into `evaluation.py` following the existing AUPRO pattern (no external dependency) —
see [Caveats](#caveats).

Every metric here is captured **on every run row**, accuracy and deployment cost
together — `inference_latency_ms_p50/p95`, `throughput_fps_bs1/bs16`,
`model_params_millions`, `peak_gpu_memory_mb`, `training_wall_clock_seconds` (0 for
every training-free method) sit in the same row as AUROC/F1/AUPRO. That pairing is
the point: it's what the dashboard's Deployment Recommender and Pareto frontier are
built on.

---

## Export to production

```bash
python training/benchmark/export.py --from-leaderboard
```

Re-fits the winner, calibrates its raw score against **held-out normal images**
(never the labeled test split, so the threshold does not leak the evaluation set),
and writes one ONNX graph matching the contract `ThresholdAnomalyDetector` already
consumes: input `"input"` of static shape `(3, H, W)` in `[0, 255]`, output
`"output"`, a scalar in `[0, 1]`. Then verifies it through the real
`OnnxInferenceEngine` path.

PatchCore / PaDiM / DFM export as a single graph. Dinomaly and the Anomalib models
do not — they go through Anomalib's own exporter, which produces a different
contract (batched NCHW in, anomaly map + score out). For Dinomaly, the `checkpoints/`
artifact above is the only persistence path available today.

---

## Design notes

**Fitting never sees a label.** Every method is fitted on normal images only, and
thresholds are chosen on held-out normal data.

**Every method sees the same geometry.** Input sizes follow each corpus's real aspect
ratio — Severstal strips are ~6:1, Kolektor panels ~2.8:1 — and squashing those to a
square would deform the defects being detected. Every method gets the same
`(height, width)` per config, so the comparison measures the method, not the resize.
Dinomaly is the documented exception: it always runs at its own fixed 392px
(`DinomalyScorer.input_size`), the scale its encoder was pretrained for, regardless of
the dataset's declared geometry — the cost pass and the pixel-metrics mask resize both
read this off the scorer rather than assuming the dataset's default.

**DINOv2 inputs are upscaled 2×.** A patch-14 ViT at 256px yields an 18×18 grid
against the CNNs' 32×32. Without the upscale the sweep would be measuring the ViT's
patch stride rather than its features.

**Cross-regime comparisons use shared configurations only.** Multi-class is undefined
for a single-category corpus, so it covers fewer configs than one-class does.
Comparing those means directly would attribute to the regime a difference that is
partly just a different set of categories.

**Cross-dataset AP/F1 comparisons are restricted, AUROC/PG2/PB2 are not.** Severstal's
~92% positive test split makes AP/F1 incomparable to the other, much-lower-prevalence
corpora — see the Severstal caveat below. Any leaderboard or dashboard view spanning
Severstal plus another dataset automatically hides `mean_ap`/`mean_f1`; AUROC and
PG2/PB2 are less prevalence-sensitive and stay.

**Metrics are computed here, not read from any backend.** Both halves of the zoo go
through `evaluation.py` on raw scores, or the two would not be comparable.

**Ensembles fuse on ranks, not raw scores, and stay single-seed.** Mahalanobis
distances and cosine errors differ by orders of magnitude; averaging them directly
would return whichever member has the biggest numbers. With 3 seeds now saving 3
separate artifacts, ensembling deliberately fuses only the lowest-seed artifact per
member rather than combinatorially fusing across seeds too.

<a id="caveats"></a>
### Caveats

- **MVTec LOCO folder layout is unverified.** The loader (`data.py`'s
  `_root_for`/`loco_defect_kind`, `datasets.py`'s `mvtec_loco_paths`, `masks.py`'s
  `_mvtec_loco_mask`) is written against the documented archive convention
  (`test/good` + `test/structural_anomalies` + `test/logical_anomalies`,
  `ground_truth/<defect>/<image>/*.png` as one mask per component) but has not been
  checked against a real download. Each of those functions is deliberately isolated
  so a post-download correction is a small, localized diff.
- **PG2/PB2's exact operating-point definition is a best-effort translation**, not
  yet cross-checked against Baitieva et al. 2025's own formulas line-by-line — see
  `evaluation.py`'s `_pg_at_bad_rate`/`_pb_at_good_rate` docstrings.
- **AUPIMO is a vendored port**, not verified bit-exact against the official
  `jpcbertoldo/aupimo` reference implementation's numeric output — only against
  hand-constructed bound cases (`tests/test_evaluation.py`).
- **Few-shot is single-seed**, not 3×, to keep the pass cheap — noted on every
  few-shot row via `single_seed=True`.
- **A crash mid-multi-class-family re-fits the whole family on resume**, not just the
  still-incomplete categories — already-`"ok"` sibling categories are harmlessly
  overwritten with the same result via idempotent update, but the GPU time to re-fit
  isn't saved. Sub-family resumability isn't implemented.
- **No drift/robustness experiment exists in this harness.** The dashboard's
  robustness section is a stub noting this — future work, not a bug.
- Anomalib models use their own input geometry and do not see `DEFAULT_SIZE`. Their
  numbers are comparable to each other and broadly to the native half, but a gap on
  the non-square datasets may be geometry rather than method.
- Anomalib models and Dinomaly fit by gradient descent, so unlike the training-free
  half their results depend on epoch count and seed.
- Severstal has no labeled-normal test split, so `datasets.py` holds out 10% of the
  normal images — and that held-out "normal" pool is Kaggle-competition-labeled and
  may itself be contaminated (`label_noise_caveat=True` on every Severstal row). Its
  positive rate is ~92%, far from the others — `test_prevalence` is stored on every
  row so this is visible directly rather than requiring a mental note. Pass
  `--severstal-target-prevalence 0.3`-ish to instead downsample its anomalous images
  to a comparable rate (optional; restricting AP/F1 from cross-dataset views is the
  default fix — see Design notes above).
- MVTec/VisA are largely saturated by 2025-era methods (Bertoldo et al., BMVC 2024;
  Heckler-Kram et al., 2025) — a near-perfect score on either says less than it used
  to. MVTec LOCO, once downloaded, is the harder comparison; the dashboard's Dataset
  Difficulty panel states this explicitly rather than burying it in a table.

---

## References

- Dinomaly — [CVPR 2025](https://arxiv.org/abs/2405.14325) · [code](https://github.com/guojiajeremy/Dinomaly) (Table 2: one-class vs multi-class gap, cited above for why one-class is skipped here)
- PatchCore — [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
- PaDiM — [Patch Distribution Modeling](https://arxiv.org/abs/2011.08785)
- AUPIMO — Bertoldo et al., "AUPIMO: Redefining PRO for Sparse Anomaly Localization," BMVC 2024 · [code](https://github.com/jpcbertoldo/aupimo)
- PG2/PB2 — Baitieva et al., "Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection," 2025
- Heckler-Kram et al., "The MVTec AD 2 Dataset," 2025 (dataset saturation)
- MVTec LOCO — Bergmann et al., "Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization," IJCV 2022
- [Anomalib](https://github.com/open-edge-platform/anomalib) · [awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
