"""Smoke tests for the top-level :mod:`adaptivevision` package."""

from __future__ import annotations

import adaptivevision


def test_package_exposes_version() -> None:
    assert isinstance(adaptivevision.__version__, str)
    assert adaptivevision.__version__


def test_version_is_in_all() -> None:
    assert "__version__" in adaptivevision.__all__
