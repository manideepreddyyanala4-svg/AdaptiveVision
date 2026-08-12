"""Golden-reference artifact loading (Milestone M6)."""

from __future__ import annotations

import json
from pathlib import Path

from adaptivevision.alignment.model import GoldenReference
from adaptivevision.common.errors import FaultError


def load_golden_reference(path: str | Path) -> GoldenReference:
    """Load a golden-reference artifact from JSON.

    Args:
        path: Filesystem path to the reference JSON document.

    Returns:
        The validated golden reference.

    Raises:
        FaultError: If the artifact cannot be read, parsed, or validated.
    """
    reference_path = Path(path)
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to load golden reference {reference_path}: {exc}"
        raise FaultError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Golden reference {reference_path} must contain a JSON object"
        raise FaultError(msg)
    try:
        return GoldenReference.from_dict(payload)
    except (KeyError, TypeError, ValueError, FaultError) as exc:
        msg = f"Invalid golden reference {reference_path}: {exc}"
        raise FaultError(msg) from exc
