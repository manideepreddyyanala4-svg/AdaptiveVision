"""AdaptiveVision - Industrial 2D/3D Vision, AOI & Metrology Edge Platform.

Top-level package. The system design is frozen in *Architecture Specification
v1.0* and the build plan in *Implementation Roadmap v1.0*. This module exposes
only the package version at Milestone M0.
"""

from __future__ import annotations

from adaptivevision import logging_setup

__all__ = ["__version__", "logging_setup"]

#: Semantic version of the platform. Kept in sync with ``pyproject.toml``.
#: A metadata-driven version is deferred until packaging matures.
__version__ = "0.0.0"
