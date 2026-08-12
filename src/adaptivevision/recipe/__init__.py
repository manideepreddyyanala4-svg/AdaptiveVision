"""Product recipe schema, storage, and lifecycle (Milestone M2).

This package defines the :class:`Recipe` aggregate - the immutable specification
for inspecting one product variant - together with its :class:`DecisionPolicy`
and the JSON-backed :class:`JsonRecipeStore` that persists and loads recipes.
The store binds the generic
:class:`~adaptivevision.common.interfaces.RecipeStore` seam to the
:class:`Recipe` aggregate.
"""

from __future__ import annotations

from adaptivevision.recipe.model import (
    DecisionPolicy,
    InspectorRegistry,
    Recipe,
    validate_inspectors,
)
from adaptivevision.recipe.store import JsonRecipeStore

__all__ = [
    "DecisionPolicy",
    "InspectorRegistry",
    "JsonRecipeStore",
    "Recipe",
    "validate_inspectors",
]
