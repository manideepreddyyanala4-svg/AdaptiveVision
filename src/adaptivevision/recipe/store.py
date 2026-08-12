"""Recipe storage and lifecycle (Milestone M2).

This module provides a concrete :class:`RecipeStore` implementation that
persists :class:`Recipe` objects as JSON files in a directory. It binds the
generic :class:`~adaptivevision.common.interfaces.RecipeStore` seam to the
:class:`Recipe` aggregate defined in this package.

Storage failures surface as :class:`~adaptivevision.common.errors.RecipeError`,
which is non-recoverable and drives the station to a fault / safe state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptivevision.common.errors import RecipeError
from adaptivevision.common.interfaces import RecipeStore
from adaptivevision.recipe.model import Recipe

#: File extension used for stored recipes.
_RECIPE_SUFFIX = ".json"


class JsonRecipeStore(RecipeStore[Recipe]):
    """A :class:`RecipeStore` backed by JSON files on disk.

    Each recipe is stored as ``<recipe_id>.json`` in the configured directory.
    The store is not thread-safe; the orchestration layer serializes access.

    Args:
        directory: Directory in which recipe files are stored. Created on first
            write if it does not exist.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize the store with a backing directory."""
        self._directory = directory

    def load(self, recipe_id: str) -> Recipe:
        """Load a recipe by identifier.

        Args:
            recipe_id: Identifier of the recipe to load.

        Returns:
            The loaded :class:`Recipe`.

        Raises:
            RecipeError: If the recipe is missing or invalid.
        """
        path = self._path_for(recipe_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            msg = f"Recipe not found: {recipe_id!r}"
            raise RecipeError(msg) from exc
        except (json.JSONDecodeError, OSError) as exc:
            msg = f"Failed to read recipe {recipe_id!r}: {exc}"
            raise RecipeError(msg) from exc
        return self._from_dict(data, recipe_id)

    def save(self, recipe: Recipe) -> None:
        """Persist a recipe.

        Args:
            recipe: The recipe to persist.

        Raises:
            RecipeError: On storage failure.
        """
        path = self._path_for(recipe.recipe_id)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(recipe.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            msg = f"Failed to write recipe {recipe.recipe_id!r}: {exc}"
            raise RecipeError(msg) from exc

    def list_ids(self) -> tuple[str, ...]:
        """Return the identifiers of all stored recipes.

        Returns:
            A tuple of recipe identifiers, sorted lexically.
        """
        if not self._directory.exists():
            return ()
        ids: list[str] = []
        for path in self._directory.glob(f"*{_RECIPE_SUFFIX}"):
            ids.append(path.stem)
        return tuple(sorted(ids))

    def _path_for(self, recipe_id: str) -> Path:
        """Return the file path for ``recipe_id``."""
        return self._directory / f"{recipe_id}{_RECIPE_SUFFIX}"

    @staticmethod
    def _from_dict(data: dict[str, Any], recipe_id: str) -> Recipe:
        """Deserialize a recipe, verifying the id matches the file name."""
        try:
            recipe = Recipe.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"Invalid recipe {recipe_id!r}: {exc}"
            raise RecipeError(msg) from exc
        if recipe.recipe_id != recipe_id:
            msg = f"Recipe id mismatch: file {recipe_id!r}, content {recipe.recipe_id!r}"
            raise RecipeError(msg)
        return recipe
