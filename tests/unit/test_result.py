"""Unit tests for :mod:`adaptivevision.common`."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from adaptivevision import common as result
from adaptivevision import common as types
from adaptivevision.common import DefectClass, Severity, Verdict


def _defect() -> result.Defect:
    return result.Defect(
        defect_class=DefectClass.SCRATCH,
        severity=Severity.MAJOR,
        score=0.87,
        roi=types.ROI(label="r", x=0.0, y=0.0, width=1.0, height=1.0),
        description="hairline scratch",
    )


def _measurement() -> types.Measurement:
    return types.Measurement(name="width", value=10.1, unit="mm")


def _defect_measurement() -> result.DefectMeasurement:
    return result.DefectMeasurement(
        bbox=(1, 2, 3, 4), area_px2=12, area_um2=48.0, aspect_ratio=1.33, morphology="particle"
    )


def test_defect_roundtrip_full_and_minimal() -> None:
    full = _defect()
    minimal = result.Defect(defect_class=DefectClass.ANOMALY, severity=Severity.MINOR)
    assert result.Defect.from_dict(full.to_dict()) == full
    assert result.Defect.from_dict(minimal.to_dict()) == minimal


def test_metrology_result_roundtrip() -> None:
    partial = result.MetrologyResult(
        measurements=(_measurement(),), defects=(_defect(),)
    )
    assert result.MetrologyResult.from_dict(partial.to_dict()) == partial


def test_anomaly_result_roundtrip() -> None:
    partial = result.AnomalyResult(
        score=0.9,
        threshold=0.5,
        is_anomalous=True,
        heatmap_ref="img/heatmap-1.png",
        defects=(_defect(),),
    )
    assert result.AnomalyResult.from_dict(partial.to_dict()) == partial


def test_classical_result_roundtrip() -> None:
    partial = result.ClassicalResult(defects=(_defect(),))
    assert result.ClassicalResult.from_dict(partial.to_dict()) == partial


def test_partial_results_are_partial_result_subtypes() -> None:
    for cls in (result.MetrologyResult, result.AnomalyResult, result.ClassicalResult):
        assert issubclass(cls, result.PartialResult)


def test_partial_result_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        result.PartialResult()  # type: ignore[abstract]


def test_inspection_result_lossless_roundtrip() -> None:
    original = result.InspectionResult(
        inspection_id="insp-1",
        part_id="part-1",
        station_id="station-A",
        verdict=Verdict.FAIL,
        recipe_ver="recipe-3",
        model_ver="model-2",
        calib_ver="calib-1",
        cycle_time_ms=142.5,
        timestamp_utc=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        measurements=(_measurement(),),
        defects=(_defect(),),
        anomaly_score=0.91,
        image_refs=("img/raw-1.png", "img/overlay-1.png"),
        defect_measurements=(_defect_measurement(),),
        drift_status="NOMINAL",
    )
    restored = result.InspectionResult.from_dict(original.to_dict())
    assert restored == original


def test_inspection_result_json_serializable() -> None:
    import json

    original = result.InspectionResult(
        inspection_id="insp-2",
        part_id="part-2",
        station_id="station-A",
        verdict=Verdict.PASS,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=100.0,
        timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC),
    )
    payload = json.dumps(original.to_dict())
    restored = result.InspectionResult.from_dict(json.loads(payload))
    assert restored == original


def test_inspection_result_is_frozen() -> None:
    res = result.InspectionResult(
        inspection_id="insp-3",
        part_id="part-3",
        station_id="s",
        verdict=Verdict.REVIEW,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=1.0,
        timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.verdict = Verdict.PASS  # type: ignore[misc]


def _retrieval_match() -> result.RetrievalMatch:
    return result.RetrievalMatch(
        vector_id=7,
        distance=0.12,
        dataset="mvtec_ad",
        category="bottle",
        defect_type="crack",
        image_path="img/ref-7.png",
        metadata={"seed": 1},
    )


def test_retrieval_match_roundtrip_full_and_minimal() -> None:
    full = _retrieval_match()
    minimal = result.RetrievalMatch(
        vector_id=0, distance=0.0, dataset="d", category="c", defect_type="t"
    )
    assert result.RetrievalMatch.from_dict(full.to_dict()) == full
    assert result.RetrievalMatch.from_dict(minimal.to_dict()) == minimal


def test_inspection_evidence_roundtrip_with_heatmap_region_and_matches() -> None:
    full = result.InspectionEvidence(
        sample_id="insp-1",
        category="bottle",
        anomaly_score=0.93,
        severity=Severity.MAJOR,
        model_ver="patchcore-v1",
        retrieval_matches=(_retrieval_match(),),
        heatmap_region="upper-right",
    )
    assert result.InspectionEvidence.from_dict(full.to_dict()) == full


def test_inspection_evidence_roundtrip_minimal_without_heatmap_region() -> None:
    minimal = result.InspectionEvidence(
        sample_id="insp-2",
        category="bottle",
        anomaly_score=None,
        severity=Severity.INFO,
        model_ver="patchcore-v1",
    )
    restored = result.InspectionEvidence.from_dict(minimal.to_dict())
    assert restored == minimal
    assert restored.heatmap_region is None


def test_advisory_report_rejects_confidence_score_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence_score"):
        result.AdvisoryReport(
            defect_classification="x",
            severity=Severity.MINOR,
            confidence_score=1.5,
            root_cause_hypothesis="h",
        )
