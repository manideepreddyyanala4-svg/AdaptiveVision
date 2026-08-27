"""HTTP route handlers for advisory reports (Milestone M19).

Read-only, mirroring :mod:`adaptivevision.api.routers.results`'s pattern: an
advisory report is produced and persisted off the request path (by whatever
calls :func:`adaptivevision.advisory.pipeline.advise` and an
:class:`~adaptivevision.common.interfaces.AdvisoryRepository`); this router
only exposes what is already stored.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from adaptivevision.common.interfaces import AdvisoryRepository

router = APIRouter(prefix="/api/v1/advisory", tags=["advisory"])


def get_advisory_repository() -> AdvisoryRepository:
    """Return the advisory repository backing the advisory endpoints.

    The concrete repository is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@router.get("/{inspection_id}")
def get_advisory_report(
    inspection_id: str,
    repository: Annotated[AdvisoryRepository, Depends(get_advisory_repository)],
) -> dict[str, object]:
    """Return the advisory report for ``inspection_id``, if one was produced."""
    report = repository.get_report(inspection_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No advisory report for inspection {inspection_id}",
        )
    return report.to_dict()
