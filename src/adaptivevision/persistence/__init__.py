"""Result persistence, image archival, traceability (Milestone M4).

This package implements the local edge persistence layer:

* :mod:`adaptivevision.persistence.database` - SQLite engine / schema setup,
* :mod:`adaptivevision.persistence.models_orm` - SQLAlchemy ORM models,
* :mod:`adaptivevision.persistence.repositories` - the
  :class:`~adaptivevision.common.interfaces.ResultRepository` implementation,
* :mod:`adaptivevision.persistence.traceability` - inspection lineage records,
* :mod:`adaptivevision.persistence.image_store` - bounded local image archival.
"""

from __future__ import annotations

from adaptivevision.persistence.database import (
    build_engine,
    init_db,
    make_session_factory,
    open_database,
    session_scope,
)
from adaptivevision.persistence.image_store import ImageStoreError, LocalImageStore
from adaptivevision.persistence.models_orm import Base, InspectionRecord
from adaptivevision.persistence.repositories import SqliteResultRepository
from adaptivevision.persistence.traceability import (
    build_mes_payload,
    build_traceability_record,
    serialize_mes_payload,
    serialize_traceability,
)

__all__ = [
    "Base",
    "ImageStoreError",
    "InspectionRecord",
    "LocalImageStore",
    "SqliteResultRepository",
    "build_engine",
    "build_mes_payload",
    "build_traceability_record",
    "init_db",
    "make_session_factory",
    "open_database",
    "serialize_mes_payload",
    "serialize_traceability",
    "session_scope",
]
