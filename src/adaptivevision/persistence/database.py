"""SQLite database layer for the local edge database (Milestone M4).

This module owns the SQLAlchemy engine and session factory for the local edge
database. It supports two modes:

* a normal, file-backed SQLite database for local operation, and
* an in-memory SQLite database (``sqlite:///:memory:``) for tests.

Schema initialization is idempotent: :func:`init_db` creates any missing tables
via the ORM metadata and is safe to call repeatedly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from adaptivevision.persistence.models_orm import Base

#: Default SQLite URL used when no explicit path is provided.
_DEFAULT_DB_PATH = "adaptivevision.db"


def build_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the local edge database.

    Args:
        url: SQLAlchemy database URL. Defaults to a file-backed SQLite database
            named :data:`_DEFAULT_DB_PATH` in the current directory. Pass
            ``"sqlite:///:memory:"`` for an in-memory database (tests).

    Returns:
        A configured :class:`~sqlalchemy.Engine`.
    """
    if url is None:
        url = f"sqlite:///{_DEFAULT_DB_PATH}"
    return create_engine(url, future=True)


def init_db(engine: Engine) -> None:
    """Create all tables defined by the ORM metadata.

    Idempotent: existing tables are left untouched.

    Args:
        engine: The engine to initialize the schema on.
    """
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a configured session factory bound to ``engine``.

    Args:
        engine: The engine the sessions will use.

    Returns:
        A :class:`~sqlalchemy.orm.sessionmaker` producing
        :class:`~sqlalchemy.orm.Session` objects.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def open_database(
    path: str | Path | None = None,
) -> tuple[Engine, sessionmaker[Session]]:
    """Open (and initialize) the local edge database.

    Args:
        path: Optional filesystem path for the SQLite file. When ``None``, the
            default file-backed database is used. Pass ``":memory:"`` for an
            in-memory database.

    Returns:
        A tuple of ``(engine, session_factory)`` with the schema initialized.
    """
    if path == ":memory:":
        url = "sqlite:///:memory:"
    elif path is None:
        url = None
    else:
        url = f"sqlite:///{Path(path)}"
    engine = build_engine(url)
    init_db(engine)
    return engine, make_session_factory(engine)


@contextlib.contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional session scope.

    Yields a session that is committed on success and rolled back on error.

    Args:
        session_factory: The session factory to create sessions from.

    Yields:
        A :class:`~sqlalchemy.orm.Session` within a transaction.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
