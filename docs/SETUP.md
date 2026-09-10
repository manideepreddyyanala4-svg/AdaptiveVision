# Setup

Python 3.11 is required (`pyproject.toml` sets `requires-python = ">=3.11"`; `ruff` and `mypy`
target `py311`).

There are two dependency sets because the runtime application and the research code have different
needs. On the development machine they live in one conda environment, but they can be installed
separately.

## Runtime environment

Enough to run the inspection application, the dashboard, and the full `tests/` suite.

```bash
python -m venv .venv && source .venv/bin/activate      # or conda create -n adaptivevision python=3.11
pip install -e ".[dev,inference,intelligence]"
```

What each group adds:

| Group | Packages | Needed for |
|---|---|---|
| (base) | numpy, sqlalchemy, fastapi, uvicorn, pydantic, PyYAML | the application itself |
| `dev` | ruff, black, mypy, pytest, pytest-cov, pre-commit, httpx, opencv-python-headless, types-PyYAML | tests, type checking, linting |
| `inference` | onnxruntime, python-multipart | real ONNX inference; the upload route |
| `intelligence` | faiss-cpu, ollama (client library) | FAISS retrieval; the LLM advisory client |

Note: `black` and `ruff` are pinned in `.pre-commit-config.yaml` (ruff 0.6.9, black 24.x). With
newer versions installed, `ruff check` and `mypy` still pass but `ruff format --check` / `black
--check` report formatting diffs that are formatter-version drift, not code problems. Use the pinned
versions if you want a clean `make check`.

## Research / training environment

Only needed for `training/` (fitting models, running the benchmark, exporting ONNX).

```bash
pip install -r training/requirements.txt
```

This adds torch, onnx, onnxruntime, tqdm, timm, scikit-learn, scipy, pandas, tabulate. Optionally
`pip install anomalib` afterward to add ~25 delegated methods to the benchmark (install it after
torch so pip does not replace a CUDA build). A GPU is needed for Dinomaly and speeds up everything
else; the training-free methods (PatchCore / PaDiM / DFM) run on CPU, just slowly.

## External assets not in this repository

| Asset | Size | How to get it |
|---|---|---|
| Datasets (MVTec AD, VisA, KolektorSDD2, Severstal) | multi-GB | download from the official sources listed in the README; put them next to the repo or set `ADAPTIVEVISION_DATA_ROOT` |
| Exported ONNX models (`models/*.onnx`) | ~14 GB total, ~96 MB each | regenerate with `python -m training.export export --method <name> --dataset <dataset>/<category>` (needs the research env + the dataset) |
| Normal-patch bank (`models/*.bank.npz`) | a few MB | `python scripts/build_patch_bank.py --images <good frames dir> --model <name>.onnx --output models/<name>.bank.npz` |
| Ollama + `qwen2.5:7b` | ~4.7 GB | `ollama serve` then `ollama pull qwen2.5:7b` (only for a real advisory; a deterministic fallback runs without it) |

The `models/*.json` manifests are committed so you can see which method/dataset/backbone each export
corresponds to and its calibration constants.

## Configuration

```bash
cp .env.example .env      # then edit
```

All settings are `ADAPTIVEVISION_*` environment variables, read by
`adaptivevision.config.load_config()`. Real environment variables take precedence over `.env`.
`configs/config.yaml` holds the metrology / drift / KPI thresholds; `recipes/demo-bottle.json` is
one product recipe (anomaly threshold and decision policy). None of these contain machine-specific
values.

## Running things

```bash
# Boot the station, run one inspection, shut down.
python scripts/run_station.py

# Serve the dashboard + REST API.
python run_dashboard.py            # http://127.0.0.1:8000/
python scripts/run_api.py          # same, without the teach-mode panel

# Score one image against a model against a labelled folder (escape / overkill sweep).
python scripts/evaluate_kpis.py --config mvtec/bottle --dataset-root <path>/mvtec/bottle

# Local Modbus TCP simulator, so the reject handshake can be exercised with no PLC.
python scripts/mock_plc.py --port 5020

# Tests.
pytest -q
mypy
ruff check .
```

With no `.env` and no model, `run_station.py` still boots a walking skeleton (null camera, no
model) and passes every part. Real inference needs `ADAPTIVEVISION_MODEL_PATH` pointing at an
exported `.onnx`, and for the RGB models `ADAPTIVEVISION_DEMO_IMAGE_PATH` pointing at a real image.

## What a fresh clone can and cannot do

| Works from a clean clone (after `pip install -e ".[dev,inference,intelligence]"`) | Needs an external asset |
|---|---|
| `pytest` (the 3 real-model e2e tests skip automatically) | real-model e2e tests -> an exported `.onnx` + MVTec bottle images |
| `mypy`, `ruff check` | — |
| `python scripts/run_station.py` (walking skeleton, PASS) | a real verdict -> an exported `.onnx` (+ image for RGB models) |
| `python run_dashboard.py` (serves empty history) | populated history -> run `run_station.py` a few times first |
| the deterministic advisory fallback | a real LLM report -> a running Ollama server |
| the whole `training/` benchmark | -> the research dependencies + the datasets |
