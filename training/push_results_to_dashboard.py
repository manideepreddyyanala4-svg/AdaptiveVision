"""Run trained PaDiM ONNX models over real test images and save results.

This does NOT re-implement inference: it uses the project's own
``OnnxInferenceEngine`` and ``ThresholdAnomalyDetector`` exactly as the
production pipeline would, then persists an ``InspectionResult`` per image via
``SqliteResultRepository`` so they show up in the dashboard
(``training/dashboard_app.py``) without inventing any new UI logic here.

Usage:
    python -m training.push_results_to_dashboard
    python training/dashboard_app.py
    # open http://127.0.0.1:8010/
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from adaptivevision.common import (
    ExecutionProvider,
    InspectionResult,
    RectifiedFrame,
    Verdict,
    new_inspection_id,
    new_part_id,
)
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import ThresholdAnomalyDetector
from adaptivevision.storage import SqliteResultRepository, open_database
from training.data import load_rgb
from training.legacy import resolve_paths

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_PER_MODEL = 24


def load_manifest(model_path: Path) -> dict:
    """Load a PaDiM model's ``<name>.json`` sidecar (dataset/category/height/width)."""
    manifest_path = model_path.with_suffix(".json")
    if not manifest_path.exists():
        msg = f"No manifest for {model_path.name}; train it with training/train_padim.py"
        raise FileNotFoundError(msg)
    return json.loads(manifest_path.read_text())


def push_model_results(
    repository: SqliteResultRepository, model_path: Path, data_root: Path
) -> int:
    """Score a sample of real test images with one trained model and save results."""
    manifest = load_manifest(model_path)
    dataset, category = manifest["dataset"], manifest["category"]
    height, width = manifest["height"], manifest["width"]

    _, test_split = resolve_paths(dataset, category, data_root)
    if not test_split:
        print(f"  skip {dataset}/{category}: no labeled test images found")
        return 0

    normal = [p for p, anomalous in test_split if not anomalous][: IMAGES_PER_MODEL // 2]
    anomalous = [p for p, anomalous in test_split if anomalous][: IMAGES_PER_MODEL // 2]
    sample = [(p, False) for p in normal] + [(p, True) for p in anomalous]

    engine = OnnxInferenceEngine(model_dir=model_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(model_path.name)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)
    station_id = f"demo-{dataset}" + (f"-{category}" if category else "")

    saved = 0
    for path, ground_truth_anomalous in sample:
        image = load_rgb(path, height, width)
        start = time.perf_counter()
        frame = RectifiedFrame(
            image=image,
            camera_id="training-demo",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        result = detector.detect(frame)
        cycle_time_ms = (time.perf_counter() - start) * 1000
        verdict = Verdict.FAIL if result.is_anomalous else Verdict.PASS

        inspection = InspectionResult(
            inspection_id=new_inspection_id(),
            part_id=new_part_id(),
            station_id=station_id,
            verdict=verdict,
            recipe_ver=f"demo-{dataset}",
            model_ver=model_path.stem,
            calib_ver="n/a",
            cycle_time_ms=cycle_time_ms,
            timestamp_utc=datetime.now(UTC),
            defects=result.defects,
            anomaly_score=result.score,
            image_refs=(
                str(path),
                "ground_truth=anomaly" if ground_truth_anomalous else "ground_truth=normal",
            ),
        )
        repository.save_result(inspection)
        saved += 1
    print(f"  {dataset}/{category or '-'}: saved {saved} results (model={model_path.stem})")
    return saved


def main() -> None:
    """Push sample results from every manifest-described PaDiM model into the dashboard DB."""
    data_root = REPO_ROOT.parent
    _, session_factory = open_database(str(REPO_ROOT / "adaptivevision.db"))
    repository = SqliteResultRepository(session_factory)

    model_paths = sorted(
        p for p in (REPO_ROOT / "models").glob("padim_*.onnx") if p.with_suffix(".json").exists()
    )
    print(
        f"Pushing anomaly-detection results into adaptivevision.db ({len(model_paths)} models) ..."
    )
    total = 0
    for model_path in model_paths:
        total += push_model_results(repository, model_path, data_root)
    print(f"done: {total} results saved. Run `python training/dashboard_app.py`")


if __name__ == "__main__":
    main()
