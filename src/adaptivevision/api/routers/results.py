"""HTTP route handlers for inspection results (Milestone M13).

The minimal API exposes read-only access to persisted inspection results so the
dashboard and external tooling can observe station output.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from adaptivevision.common.interfaces import ResultRepository
from adaptivevision.common.result import InspectionResult

router = APIRouter(prefix="/api/v1/results", tags=["results"])


def get_repository() -> ResultRepository:
    """Return the result repository backing the results endpoints.

    The concrete repository is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


def _to_dict(result: InspectionResult) -> dict[str, object]:
    """Convert an inspection result to a JSON-serializable mapping."""
    return result.to_dict()


@router.get("")
def list_results(
    repository: Annotated[ResultRepository, Depends(get_repository)],
    limit: int = 100,
    offset: int = 0,
) -> dict[str, object]:
    """Return a page of inspection results, most-recent first."""
    results = repository.list_results(limit=limit, offset=offset)
    return {"items": [_to_dict(r) for r in results], "count": len(results)}


@router.get("/{inspection_id}")
def get_result(
    inspection_id: str,
    repository: Annotated[ResultRepository, Depends(get_repository)],
) -> dict[str, object]:
    """Return a single inspection result by identifier."""
    result = repository.get_result(inspection_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {inspection_id} not found",
        )
    return _to_dict(result)
