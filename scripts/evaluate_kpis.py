"""Industrial yield KPI evaluator: Overkill Rate vs. Escape Rate (Milestone M21).

A benchmark leaderboard ranks models by AUROC; a fab has to commit to one
threshold and live with the two errors it can make at completely different
prices. This script scores a real, labeled test set with one of this
project's 29 production ONNX models, sweeps every possible operating
threshold, and reports:

* **Escape Rate** (a.k.a. underkill / false-negative rate): defective parts
  that would be misclassified PASS -- the number a line cannot tolerate being
  far from zero.
* **Overkill Rate** (a.k.a. false-alarm / false-positive rate): good parts
  that would be misclassified FAIL -- the yield a line gives up to hold that
  escape rate down.

It then reports the *operating recommendation*: the threshold with the
lowest overkill rate that still keeps the escape rate at or below a target
(default from ``configs/config.yaml``'s ``kpi.target_escape_rate``, e.g.
0.1%).

Dataset ingestion is intentionally narrow, not a general-purpose loader: it
supports the two labeled-folder conventions this repo's own MVTec and VisA
copies actually use (``test/good/`` vs other subfolders; ``Data/Images/Normal``
vs ``Data/Images/Anomaly``). Kolektor and Severstal ship as CSV/weak-label
formats this script does not parse -- passing their dataset root raises a
clear error rather than silently guessing. This deliberately does not import
``training/benchmark``'s own richer dataset-loading code: that code is
part of this project's training-only surface (it depends on ``torch``, and
sits outside the ``mypy --strict`` tree by design -- see
``docs/milestones/M19.md``), while this script is checked the same way
``src/`` is (``pyproject.toml``'s ``[tool.mypy] files``).

Usage:
    python scripts/evaluate_kpis.py --config mvtec/bottle \\
        --dataset-root /path/to/mvtec/bottle

    python scripts/evaluate_kpis.py --config visa/candle \\
        --dataset-root /path/to/VisA_20220922/candle --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from adaptivevision.common import RectifiedFrame
from adaptivevision.config import load_aoi_config
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import ThresholdAnomalyDetector

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS_DIR = _REPO_ROOT / "models"

#: Extensions treated as test images when scanning a dataset directory.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

_NDArray = np.ndarray[Any, np.dtype[Any]]


@dataclass(frozen=True, slots=True)
class KpiResult:
    """Result of an Overkill-vs-Escape-Rate KPI evaluation.

    Attributes:
        config: Dataset/category identifier evaluated (e.g. ``"mvtec/bottle"``).
        model_name: ONNX model filename used.
        n_normal: Number of known-good test images scored.
        n_anomalous: Number of known-defective test images scored.
        target_escape_rate: The escape-rate ceiling that was searched for.
        threshold: The recommended operating threshold.
        escape_rate: Fraction of defective parts scored below ``threshold``
            (misclassified PASS) at that threshold.
        overkill_rate: Fraction of good parts scored at/above ``threshold``
            (misclassified FAIL) at that threshold.
        achieved: Whether a threshold satisfying
            ``escape_rate <= target_escape_rate`` was actually found. When
            ``False``, ``threshold`` is the strictest available (lowest
            achievable escape rate), which still exceeds the target -- this
            model/dataset cannot hit the target at any threshold.
    """

    config: str
    model_name: str
    n_normal: int
    n_anomalous: int
    target_escape_rate: float
    threshold: float
    escape_rate: float
    overkill_rate: float
    achieved: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "config": self.config,
            "model_name": self.model_name,
            "n_normal": self.n_normal,
            "n_anomalous": self.n_anomalous,
            "target_escape_rate": self.target_escape_rate,
            "threshold": self.threshold,
            "escape_rate": self.escape_rate,
            "overkill_rate": self.overkill_rate,
            "achieved": self.achieved,
        }


def find_operating_point(
    scores: _NDArray, labels: _NDArray, target_escape_rate: float
) -> tuple[float, float, float, bool]:
    """Find the highest (strictest) threshold meeting an escape-rate ceiling.

    Sweeps every distinct score in ``scores`` as a candidate threshold
    (a sample is called anomalous when its score is ``>= threshold``).
    Escape rate falls and overkill rate rises as the threshold drops, so the
    operating point that minimizes overkill subject to the escape-rate
    ceiling is the *highest* threshold whose escape rate still meets it.

    Args:
        scores: Anomaly scores, higher meaning more anomalous.
        labels: Boolean ground truth, ``True`` for a real defect.
        target_escape_rate: Maximum acceptable escape rate.

    Returns:
        ``(threshold, escape_rate, overkill_rate, achieved)``. When no
        threshold meets the target, returns the strictest threshold (the
        lowest achievable escape rate) with ``achieved=False``.

    Raises:
        ValueError: If ``scores`` has no normal or no anomalous samples --
            escape/overkill rate are both undefined without both classes.
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        msg = (
            "find_operating_point requires at least one normal and one "
            f"anomalous sample; got {n_neg} normal, {n_pos} anomalous"
        )
        raise ValueError(msg)

    # Highest threshold first: strictest (fewest false alarms) to loosest.
    candidate_thresholds = np.sort(np.unique(scores))[::-1]

    best: tuple[float, float, float] | None = None
    strictest: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds:
        predicted_anomalous = scores >= threshold
        escapes = int(np.sum(~predicted_anomalous & labels))
        overkills = int(np.sum(predicted_anomalous & ~labels))
        escape_rate = escapes / n_pos
        overkill_rate = overkills / n_neg

        if strictest is None:
            strictest = (float(threshold), escape_rate, overkill_rate)
        if escape_rate <= target_escape_rate:
            best = (float(threshold), escape_rate, overkill_rate)
            break

    if best is not None:
        return (*best, True)
    assert strictest is not None  # candidate_thresholds is non-empty: n_pos, n_neg > 0
    return (*strictest, False)


def _discover_labeled_images(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """Discover (normal_images, defective_images) under ``dataset_root``.

    Tries this repo's two known real conventions in order:

    * MVTec-style: ``dataset_root/test/good/*`` vs every other
      ``dataset_root/test/<defect_type>/*``.
    * VisA-style: ``dataset_root/Data/Images/Normal/*`` vs
      ``dataset_root/Data/Images/Anomaly/*``.

    Args:
        dataset_root: Root directory of one dataset category.

    Returns:
        ``(normal_images, defective_images)``, both sorted for determinism.

    Raises:
        FileNotFoundError: If neither convention matches -- most likely
            Kolektor or Severstal, whose CSV/weak-label formats this
            function does not parse.
    """
    mvtec_test = dataset_root / "test"
    mvtec_good = mvtec_test / "good"
    if mvtec_good.is_dir():
        normal = _list_images(mvtec_good)
        defective: list[Path] = []
        for child in sorted(mvtec_test.iterdir()):
            if child.is_dir() and child.name != "good":
                defective.extend(_list_images(child))
        return normal, defective

    visa_normal = dataset_root / "Data" / "Images" / "Normal"
    visa_anomaly = dataset_root / "Data" / "Images" / "Anomaly"
    if visa_normal.is_dir() and visa_anomaly.is_dir():
        return _list_images(visa_normal), _list_images(visa_anomaly)

    msg = (
        f"Could not find a recognized labeled-image layout under {dataset_root} "
        "(expected test/good/ + test/<defect>/, MVTec-style, or "
        "Data/Images/Normal/ + Data/Images/Anomaly/, VisA-style). "
        "Kolektor and Severstal ship as CSV/weak-label formats this script "
        "does not ingest."
    )
    raise FileNotFoundError(msg)


def _list_images(directory: Path) -> list[Path]:
    """Return every image file directly inside ``directory``, sorted."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def score_images(
    engine: OnnxInferenceEngine, manifest: dict[str, Any], images: list[Path]
) -> _NDArray:
    """Score every image in ``images`` with ``engine``, one score each.

    Reuses :class:`ThresholdAnomalyDetector` (rather than calling
    ``engine.infer`` directly) so image resizing and the channel-last ->
    channel-first conversion go through the same, already-tested path
    production inference uses -- this project has twice shipped a real bug
    from re-deriving that transpose by hand (see ``docs/milestones/M20.md``).

    Args:
        engine: A loaded :class:`OnnxInferenceEngine`.
        manifest: The model's manifest dict (needs ``"height"``/``"width"``).
        images: Image file paths to score.

    Returns:
        One anomaly score per image, in ``images`` order.

    Raises:
        ValueError: If an image file fails to decode.
    """
    # threshold=0.0 is arbitrary and unused here: only .score is read below,
    # never .is_anomalous, so the configured decision boundary is irrelevant.
    detector = ThresholdAnomalyDetector(engine, threshold=0.0)
    height, width = int(manifest["height"]), int(manifest["width"])

    scores = np.empty(len(images), dtype=np.float64)
    for index, image_path in enumerate(images):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f"Could not decode image: {image_path}"
            raise ValueError(msg)
        resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = RectifiedFrame(
            image=rgb,
            camera_id="evaluate-kpis",
            frame_id=image_path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        scores[index] = detector.detect(frame).score
    return scores


def _resolve_model(config: str, models_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Find the production ONNX model and manifest for ``config``.

    Args:
        config: Dataset/category identifier, e.g. ``"mvtec/bottle"`` or
            ``"kolektor"`` (no category).
        models_dir: Directory of exported ``.onnx`` + ``.json`` pairs.

    Returns:
        ``(onnx_path, manifest)``.

    Raises:
        FileNotFoundError: If no manifest in ``models_dir`` matches ``config``.
    """
    dataset, _, category = config.partition("/")
    for manifest_path in sorted(models_dir.glob("*.json")):
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_config = manifest.get("dataset", "")
        if manifest.get("category"):
            manifest_config = f"{manifest_config}/{manifest['category']}"
        if manifest_config == config or (not category and manifest.get("dataset") == dataset):
            onnx_path = manifest_path.with_suffix(".onnx")
            if onnx_path.is_file():
                return onnx_path, manifest
    msg = f"No exported model found for config {config!r} under {models_dir}"
    raise FileNotFoundError(msg)


def evaluate(
    config: str,
    dataset_root: Path,
    *,
    models_dir: Path = _DEFAULT_MODELS_DIR,
    target_escape_rate: float,
) -> KpiResult:
    """Run the full KPI evaluation for one dataset/category.

    Args:
        config: Dataset/category identifier, e.g. ``"mvtec/bottle"``.
        dataset_root: Root directory of that dataset's real test images.
        models_dir: Directory of exported production models.
        target_escape_rate: Maximum acceptable escape rate.

    Returns:
        The populated :class:`KpiResult`.
    """
    onnx_path, manifest = _resolve_model(config, models_dir)
    normal_images, defective_images = _discover_labeled_images(dataset_root)

    engine = OnnxInferenceEngine(model_dir=onnx_path.parent)
    engine.load(onnx_path.name)
    engine.warmup()

    normal_scores = score_images(engine, manifest, normal_images)
    defective_scores = score_images(engine, manifest, defective_images)

    scores = np.concatenate([normal_scores, defective_scores])
    labels = np.concatenate(
        [np.zeros(len(normal_scores), dtype=bool), np.ones(len(defective_scores), dtype=bool)]
    )
    threshold, escape_rate, overkill_rate, achieved = find_operating_point(
        scores, labels, target_escape_rate
    )

    return KpiResult(
        config=config,
        model_name=onnx_path.name,
        n_normal=len(normal_images),
        n_anomalous=len(defective_images),
        target_escape_rate=target_escape_rate,
        threshold=threshold,
        escape_rate=escape_rate,
        overkill_rate=overkill_rate,
        achieved=achieved,
    )


def _format_report(result: KpiResult) -> str:
    """Render a :class:`KpiResult` as a human-readable operator report."""
    status = "TARGET MET" if result.achieved else "TARGET NOT MET at any threshold"
    return (
        f"KPI evaluation: {result.config} ({result.model_name})\n"
        f"  test images: {result.n_normal} normal, {result.n_anomalous} anomalous\n"
        f"  target escape rate: <= {result.target_escape_rate:.4%}\n"
        f"  --- recommended operating point ---\n"
        f"  threshold:      {result.threshold:.6f}\n"
        f"  escape rate:    {result.escape_rate:.4%}  (defects misclassified PASS)\n"
        f"  overkill rate:  {result.overkill_rate:.4%}  (good parts misclassified FAIL)\n"
        f"  status: {status}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code: ``0`` if the target escape rate was achieved at
        some threshold, ``1`` otherwise (including dataset/model resolution
        failures) -- so this composes into an automated deployment gate.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument(
        "--config", required=True, help="Dataset/category, e.g. mvtec/bottle, kolektor"
    )
    parser.add_argument(
        "--dataset-root", required=True, type=Path, help="Root directory of real test images"
    )
    parser.add_argument("--models-dir", type=Path, default=_DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--target-escape-rate",
        type=float,
        default=None,
        help="Overrides configs/config.yaml's kpi.target_escape_rate",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    args = parser.parse_args(argv)

    target = args.target_escape_rate
    if target is None:
        target = load_aoi_config().kpi.target_escape_rate

    try:
        result = evaluate(
            args.config,
            args.dataset_root,
            models_dir=args.models_dir,
            target_escape_rate=target,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2) if args.json else _format_report(result))
    return 0 if result.achieved else 1


if __name__ == "__main__":
    raise SystemExit(main())
