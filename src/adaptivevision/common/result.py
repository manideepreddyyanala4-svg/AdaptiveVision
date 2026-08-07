"""Result value objects: defects, partial results, and the aggregate result.

Per frozen decisions 4 and 5, all result types are frozen dataclasses with
explicit ``to_dict`` / ``from_dict`` serialization. :class:`InspectionResult`
round-trips losslessly, which the M1 acceptance criteria require.

This module defines only the *shapes* of results. The logic that produces them
lives in the metrology (M7), anomaly (M9), and decision (M10) milestones.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from adaptivevision.common.enums import DefectClass, Severity, Verdict
from adaptivevision.common.types import ROI, Measurement


@dataclass(frozen=True, slots=True)
class Defect:
    """A single detected defect.

    Attributes:
        defect_class: Category of the defect.
        severity: Severity ranking.
        score: Optional confidence or anomaly score in ``[0, 1]``.
        roi: Optional region the defect was localized to.
        description: Optional human-readable note.
    """

    defect_class: DefectClass
    severity: Severity
    score: float | None = None
    roi: ROI | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "defect_class": self.defect_class.value,
            "severity": self.severity.value,
            "score": self.score,
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        roi_data = data.get("roi")
        return cls(
            defect_class=DefectClass(data["defect_class"]),
            severity=Severity(data["severity"]),
            score=data.get("score"),
            roi=ROI.from_dict(roi_data) if roi_data is not None else None,
            description=data.get("description"),
        )


class PartialResult(abc.ABC):
    """Base type for the output of a single inspector."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""


@dataclass(frozen=True, slots=True)
class MetrologyResult(PartialResult):
    """Output of the metrology inspector (Milestone M7).

    Attributes:
        measurements: Measured features with recorded tolerance outcomes.
        defects: Dimensional defects raised by the inspector.
    """

    measurements: tuple[Measurement, ...] = ()
    defects: tuple[Defect, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "measurements": [m.to_dict() for m in self.measurements],
            "defects": [d.to_dict() for d in self.defects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            measurements=tuple(Measurement.from_dict(m) for m in data["measurements"]),
            defects=tuple(Defect.from_dict(d) for d in data["defects"]),
        )


@dataclass(frozen=True, slots=True)
class AnomalyResult(PartialResult):
    """Output of the anomaly inspector (Milestone M9).

    Attributes:
        score: Anomaly score for the part.
        threshold: Decision threshold the score was compared against.
        is_anomalous: Whether the detector flagged the part as anomalous.
        heatmap_ref: Optional reference to the archived anomaly heatmap.
        defects: Anomaly defects raised by the inspector.
    """

    score: float
    threshold: float
    is_anomalous: bool
    heatmap_ref: str | None = None
    defects: tuple[Defect, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "score": self.score,
            "threshold": self.threshold,
            "is_anomalous": self.is_anomalous,
            "heatmap_ref": self.heatmap_ref,
            "defects": [d.to_dict() for d in self.defects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            score=data["score"],
            threshold=data["threshold"],
            is_anomalous=data["is_anomalous"],
            heatmap_ref=data.get("heatmap_ref"),
            defects=tuple(Defect.from_dict(d) for d in data["defects"]),
        )


@dataclass(frozen=True, slots=True)
class ClassicalResult(PartialResult):
    """Output of the classical AOI inspector (Phase 3).

    Minimal at M1; extended by later milestones. Existing fields are stable.

    Attributes:
        defects: Defects raised by classical checks.
    """

    defects: tuple[Defect, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {"defects": [d.to_dict() for d in self.defects]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(defects=tuple(Defect.from_dict(d) for d in data["defects"]))


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """The complete, traceable result of inspecting one part.

    Carries the full lineage (recipe / model / calibration versions) required
    for traceability (Architecture Spec v1.0, FR-T1). Round-trips losslessly via
    :meth:`to_dict` / :meth:`from_dict`.

    Attributes:
        inspection_id: Unique identifier of this inspection.
        part_id: Identifier of the inspected part.
        station_id: Identifier of the station that produced the result.
        verdict: Final verdict.
        recipe_ver: Version of the active recipe.
        model_ver: Version of the anomaly model, if any.
        calib_ver: Version of the calibration applied.
        cycle_time_ms: End-to-end inspection time in milliseconds.
        timestamp_utc: Completion time (timezone-aware, UTC).
        measurements: Measured features.
        defects: Detected defects.
        anomaly_score: Overall anomaly score, if computed.
        image_refs: References to archived images for this part.
    """

    inspection_id: str
    part_id: str
    station_id: str
    verdict: Verdict
    recipe_ver: str
    model_ver: str
    calib_ver: str
    cycle_time_ms: float
    timestamp_utc: datetime
    measurements: tuple[Measurement, ...] = ()
    defects: tuple[Defect, ...] = ()
    anomaly_score: float | None = None
    image_refs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "inspection_id": self.inspection_id,
            "part_id": self.part_id,
            "station_id": self.station_id,
            "verdict": self.verdict.value,
            "recipe_ver": self.recipe_ver,
            "model_ver": self.model_ver,
            "calib_ver": self.calib_ver,
            "cycle_time_ms": self.cycle_time_ms,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "measurements": [m.to_dict() for m in self.measurements],
            "defects": [d.to_dict() for d in self.defects],
            "anomaly_score": self.anomaly_score,
            "image_refs": list(self.image_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            inspection_id=data["inspection_id"],
            part_id=data["part_id"],
            station_id=data["station_id"],
            verdict=Verdict(data["verdict"]),
            recipe_ver=data["recipe_ver"],
            model_ver=data["model_ver"],
            calib_ver=data["calib_ver"],
            cycle_time_ms=data["cycle_time_ms"],
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]),
            measurements=tuple(Measurement.from_dict(m) for m in data["measurements"]),
            defects=tuple(Defect.from_dict(d) for d in data["defects"]),
            anomaly_score=data.get("anomaly_score"),
            image_refs=tuple(data.get("image_refs", ())),
        )
