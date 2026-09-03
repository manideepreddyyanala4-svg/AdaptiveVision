"""Inspection traceability (Milestone M4).

Traceability preserves the full lineage of an inspection so an operator can
reconstruct exactly what happened for a given part (Architecture Spec v1.0,
FR-T1). This module builds a JSON-friendly traceability record from an
:class:`~adaptivevision.common.result.InspectionResult`, capturing:

* the inspection and part identifiers,
* the recipe / model / calibration versions,
* the verdict and cycle time,
* the start and end timestamps,
* the serialized defects and measurements / anomaly information,
* and the references to archived images.

The record is stored alongside the result in the database and is also the
canonical payload for log-based tracing.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from adaptivevision.common.result import InspectionResult


def build_traceability_record(result: InspectionResult) -> dict[str, Any]:
    """Build a JSON-friendly traceability record for an inspection result.

    Args:
        result: The completed inspection result.

    Returns:
        A dictionary capturing the full inspection lineage.
    """
    return {
        "inspection_id": result.inspection_id,
        "part_id": result.part_id,
        "station_id": result.station_id,
        "recipe_version": result.recipe_ver,
        "model_version": result.model_ver,
        "calibration_version": result.calib_ver,
        "verdict": result.verdict.value,
        "cycle_time_ms": result.cycle_time_ms,
        "start_timestamp": _iso(result.timestamp_utc),
        "end_timestamp": _iso(result.timestamp_utc),
        "defects": [d.to_dict() for d in result.defects],
        "measurements": [m.to_dict() for m in result.measurements],
        "anomaly_score": result.anomaly_score,
        "image_refs": list(result.image_refs),
    }


def serialize_traceability(result: InspectionResult) -> str:
    """Serialize a traceability record to a JSON string.

    Args:
        result: The completed inspection result.

    Returns:
        A JSON string capturing the inspection lineage.
    """
    return json.dumps(build_traceability_record(result), sort_keys=True)


def build_mes_payload(result: InspectionResult) -> dict[str, Any]:
    """Build a JSON-friendly MES (Manufacturing Execution System) event payload
    for an inspection result (Milestone M21).

    There is no single universal MES event schema -- real integrations vary
    per vendor and site (SEMI E10/E30-style equipment events, proprietary
    line-controller formats, etc.). This is a reasonable, documented default
    shape: the fields a downstream MES/SCADA system typically keys routing or
    alarm logic off (disposition, defect summary, drift status) promoted to
    the top level, with the full traceability record nested under
    ``detail`` for audit trails -- not a specific vendor integration.

    Args:
        result: The completed inspection result.

    Returns:
        A dictionary shaped as one MES inspection event.
    """
    areas = [m.area_um2 for m in result.defect_measurements]
    return {
        "event_type": "INSPECTION_RESULT",
        "event_timestamp": _iso(result.timestamp_utc),
        "station_id": result.station_id,
        "part_id": result.part_id,
        "inspection_id": result.inspection_id,
        "disposition": result.verdict.value.upper(),
        "defect_summary": {
            "defect_count": len(result.defect_measurements),
            "max_defect_area_um2": max(areas) if areas else None,
            "dominant_defect_type": (
                result.defect_measurements[0].morphology if result.defect_measurements else None
            ),
        },
        "drift_status": result.drift_status,
        "detail": build_traceability_record(result),
    }


def serialize_mes_payload(result: InspectionResult) -> str:
    """Serialize an MES event payload to a JSON string.

    Args:
        result: The completed inspection result.

    Returns:
        A JSON string of the MES event payload built by :func:`build_mes_payload`.
    """
    return json.dumps(build_mes_payload(result), sort_keys=True)


def _iso(value: datetime) -> str:
    """Return an ISO-8601 string for ``value``."""
    return value.isoformat()
