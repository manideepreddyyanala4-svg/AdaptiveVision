"""Persistence integration with the inspection flow (Milestone M4).

Persistence must stay off the inspection critical path: a database failure must
never crash or block the inspection loop. This module provides a result handler
that is wired into the scheduler's ``on_result`` callback. It persists each
completed inspection result (and archives its image references) and logs any
failure clearly, preserving the ``inspection_id`` so an operator can trace a
result from the logs to the database.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from adaptivevision.common.result import InspectionResult
from adaptivevision.persistence.image_store import LocalImageStore
from adaptivevision.persistence.repositories import SqliteResultRepository

logger = logging.getLogger("adaptivevision.persistence")


class PersistenceHandler:
    """Persists inspection results off the critical path.

    Args:
        repository: The result repository to persist to.
        image_store: Optional image store used to archive image references.
    """

    def __init__(
        self,
        repository: SqliteResultRepository,
        image_store: LocalImageStore | None = None,
    ) -> None:
        """Initialize the handler."""
        self._repository = repository
        self._image_store = image_store

    def on_result(self, result: InspectionResult) -> None:
        """Persist a completed inspection result.

        Failures are logged and swallowed so the inspection loop is never
        blocked or crashed by a persistence problem.

        Args:
            result: The completed inspection result.
        """
        try:
            self._repository.save_result(result)
        except Exception as exc:
            logger.error(
                "Failed to persist inspection result",
                extra={
                    "inspection_id": result.inspection_id,
                    "part_id": result.part_id,
                    "error": str(exc),
                },
            )
            return
        logger.info(
            "Inspection result persisted",
            extra={
                "inspection_id": result.inspection_id,
                "part_id": result.part_id,
                "verdict": result.verdict.value,
            },
        )


def make_persistence_handler(
    repository: SqliteResultRepository,
    image_store: LocalImageStore | None = None,
) -> Callable[[InspectionResult], None]:
    """Build an ``on_result`` callback that persists results.

    Args:
        repository: The result repository to persist to.
        image_store: Optional image store used to archive image references.

    Returns:
        A callable suitable for the scheduler's ``on_result`` hook.
    """
    handler = PersistenceHandler(repository, image_store)
    return handler.on_result
