"""Unit tests for :mod:`adaptivevision.common.enums`."""

from __future__ import annotations

from adaptivevision.common import enums


def test_verdict_values_are_pinned() -> None:
    assert enums.Verdict.PASS.value == "pass"
    assert enums.Verdict.FAIL.value == "fail"
    assert enums.Verdict.REVIEW.value == "review"


def test_verdict_is_string() -> None:
    assert enums.Verdict.PASS == "pass"
    assert isinstance(enums.Verdict.PASS, str)


def test_severity_values_are_pinned() -> None:
    assert [s.value for s in enums.Severity] == ["info", "minor", "major", "critical"]


def test_defect_class_values_are_pinned() -> None:
    assert enums.DefectClass.DIMENSIONAL.value == "dimensional"
    assert enums.DefectClass.ANOMALY.value == "anomaly"
    assert enums.DefectClass.UNKNOWN.value == "unknown"


def test_station_state_covers_spec_states() -> None:
    values = {s.value for s in enums.StationState}
    assert {"init", "running", "fault", "estop", "shutdown"} <= values


def test_roundtrip_from_value() -> None:
    for member in enums.CameraKind:
        assert enums.CameraKind(member.value) is member


def test_execution_provider_order() -> None:
    assert [p.value for p in enums.ExecutionProvider] == [
        "tensorrt",
        "cuda",
        "openvino",
        "cpu",
    ]
