"""Real-image proof that the wired M9/M10 pipeline gives correct verdicts.

``tests/e2e/test_run_station.py`` proves the walking skeleton boots cleanly
with *no* model configured. This test proves the opposite half of the story:
with a real ONNX model and a real camera frame, the pipeline built by
``app.app.build_station``'s helpers (``build_anomaly_detector``,
``build_decision_policy``) actually distinguishes a good part from a
defective one -- "picture in, correct yes/no out" end-to-end, not just "the
plumbing doesn't crash."

Skipped when the MVTec bottle category or the benchmark-exported ONNX model
aren't present locally (both are produced by ``training/``, not part of the
repo itself, and this machine-specific data won't exist in every checkout or
in CI).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from adaptivevision.camera import build_frame
from adaptivevision.common import ExecutionProvider, Verdict
from adaptivevision.common import CameraDriver
from adaptivevision.common import RawFrame
from adaptivevision.decision import DecisionPolicy
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import ThresholdAnomalyDetector
from adaptivevision.orchestration import InspectionPipeline
from adaptivevision.camera import resize_to

_DATA_ROOT = Path("/home/tonyai/Documents/adaptivevision_M1 (2)/adaptivevision_M1")
_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "patchcore_dinov2_vitb14__mvtec_bottle.onnx"
)
_BOTTLE_TEST_DIR = _DATA_ROOT / "mvtec" / "bottle" / "test"

pytestmark = pytest.mark.skipif(
    not (_MODEL_PATH.is_file() and _BOTTLE_TEST_DIR.is_dir()),
    reason="requires the local MVTec bottle category and the benchmark-exported ONNX model",
)


class _FixedImageCamera(CameraDriver):
    """A camera driver that always returns one real, pre-loaded image."""

    def __init__(self, image_path: Path) -> None:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f"Could not read test image: {image_path}"
            raise FileNotFoundError(msg)
        self._image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def capture(self, trigger_id: str | None = None) -> RawFrame:
        assert self._opened
        return build_frame(self._image.copy(), "test-cam", trigger_id=trigger_id)

    def is_healthy(self) -> bool:
        return self._opened


def _build_pipeline(image_path: Path) -> InspectionPipeline:
    engine = OnnxInferenceEngine(
        model_dir=str(_MODEL_PATH.parent),
        providers=(ExecutionProvider.CPU,),
    )
    engine.load(_MODEL_PATH.name)
    engine.warmup()
    # Threshold empirically verified against this exact model: good bottles
    # score 0.44-0.60, every defect category tested scores 1.0 (see
    # docs/milestones/M20.md for the verification transcript).
    detector = ThresholdAnomalyDetector(engine, threshold=0.7)
    camera = _FixedImageCamera(image_path)
    camera.open()
    return InspectionPipeline(
        camera,
        station_id="test-station",
        recipe_ver="test",
        preprocessor=resize_to(256, 256),
        anomaly_detector=detector,
        decision_policy=DecisionPolicy(),
    )


def test_good_bottle_passes() -> None:
    image_path = next((_BOTTLE_TEST_DIR / "good").glob("*.png"))
    pipeline = _build_pipeline(image_path)

    result = pipeline.run("part-good", trigger_id="t1")

    assert result.verdict == Verdict.PASS
    assert result.anomaly_score is not None
    assert result.anomaly_score < 0.7


@pytest.mark.parametrize("defect_kind", ["broken_large", "broken_small", "contamination"])
def test_defective_bottle_fails(defect_kind: str) -> None:
    defect_dir = _BOTTLE_TEST_DIR / defect_kind
    if not defect_dir.is_dir():
        pytest.skip(f"no {defect_kind} category present locally")
    image_path = next(defect_dir.glob("*.png"))
    pipeline = _build_pipeline(image_path)

    result = pipeline.run("part-defective", trigger_id="t1")

    assert result.verdict == Verdict.FAIL
    assert result.anomaly_score is not None
    assert result.anomaly_score >= 0.7
    assert len(result.defects) == 1
    assert result.defects[0].description == "Anomaly score exceeded threshold"
