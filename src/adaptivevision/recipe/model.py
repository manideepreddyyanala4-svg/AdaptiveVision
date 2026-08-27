"""The product recipe aggregate (Milestone M2).

A :class:`Recipe` is the immutable specification for inspecting one product
variant. It composes the shared domain vocabulary defined in Milestone M1:

* :class:`~adaptivevision.common.types.ROI` regions of interest,
* :class:`~adaptivevision.common.types.MeasurementSpec` / ``Tolerance`` bands,
* the decision enums (:class:`~adaptivevision.common.enums.Verdict`,
  :class:`~adaptivevision.common.enums.Severity`),
* and a set of inspector references.

Per the frozen decision recorded in the M1 notes, inspector references are
*validated strings* resolved against a registry rather than a new enum, so no
spec change is required. The registry is supplied by the caller at
construction time; the recipe itself stores only the validated names.

Invalid recipes raise :class:`~adaptivevision.common.errors.RecipeError`, which
is non-recoverable and drives the station to a fault / safe state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from adaptivevision.common.enums import Severity
from adaptivevision.common.errors import RecipeError
from adaptivevision.common.types import ROI, MeasurementSpec

#: A registry of known inspector names, keyed by the string a recipe uses.
InspectorRegistry = frozenset[str]


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Rules that map inspection outcomes to a final verdict (Milestone M10).

    At M2 this is a *declared* policy carried by the recipe; the logic that
    applies it belongs to the decision milestone (M10). The fields are the
    stable contract the decision engine will consume.

    Attributes:
        anomaly_threshold: Score above which a part is flagged anomalous.
        review_on_anomaly: Whether an anomaly forces ``REVIEW`` rather than
            ``FAIL``.
        max_defects: Maximum tolerated defect count before ``FAIL``.
        fail_severity: Minimum severity that forces ``FAIL``.
    """

    anomaly_threshold: float = 0.5
    review_on_anomaly: bool = False
    max_defects: int = 0
    fail_severity: Severity = Severity.MAJOR

    def __post_init__(self) -> None:
        """Validate policy invariants."""
        if not 0.0 <= self.anomaly_threshold <= 1.0:
            msg = "DecisionPolicy.anomaly_threshold must be in [0, 1]"
            raise RecipeError(msg)
        if self.max_defects < 0:
            msg = "DecisionPolicy.max_defects must be non-negative"
            raise RecipeError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "anomaly_threshold": self.anomaly_threshold,
            "review_on_anomaly": self.review_on_anomaly,
            "max_defects": self.max_defects,
            "fail_severity": self.fail_severity.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            anomaly_threshold=data.get("anomaly_threshold", 0.5),
            review_on_anomaly=data.get("review_on_anomaly", False),
            max_defects=data.get("max_defects", 0),
            fail_severity=Severity(data.get("fail_severity", "major")),
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    """The immutable specification for inspecting one product variant.

    Attributes:
        recipe_id: Stable identifier of the recipe.
        version: Version string of this recipe revision.
        rois: Regions of interest inspected by this recipe.
        measurement_specs: Dimensional specifications to evaluate.
        inspectors: Validated inspector names to run, in order.
        decision: Decision policy for this recipe.
        product_name: Optional human-readable product name.
    """

    recipe_id: str
    version: str
    rois: tuple[ROI, ...] = ()
    measurement_specs: tuple[MeasurementSpec, ...] = ()
    inspectors: tuple[str, ...] = ()
    decision: DecisionPolicy = field(default_factory=DecisionPolicy)
    product_name: str | None = None

    def __post_init__(self) -> None:
        """Validate recipe invariants."""
        if not self.recipe_id:
            msg = "Recipe.recipe_id must not be empty"
            raise RecipeError(msg)
        if not self.version:
            msg = "Recipe.version must not be empty"
            raise RecipeError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "recipe_id": self.recipe_id,
            "version": self.version,
            "rois": [roi.to_dict() for roi in self.rois],
            "measurement_specs": [spec.to_dict() for spec in self.measurement_specs],
            "inspectors": list(self.inspectors),
            "decision": self.decision.to_dict(),
            "product_name": self.product_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            recipe_id=data["recipe_id"],
            version=data["version"],
            rois=tuple(ROI.from_dict(r) for r in data.get("rois", ())),
            measurement_specs=tuple(
                MeasurementSpec.from_dict(s) for s in data.get("measurement_specs", ())
            ),
            inspectors=tuple(data.get("inspectors", ())),
            decision=DecisionPolicy.from_dict(data.get("decision", {})),
            product_name=data.get("product_name"),
        )


def validate_inspectors(
    inspectors: tuple[str, ...],
    registry: InspectorRegistry,
) -> tuple[str, ...]:
    """Validate inspector names against ``registry``.

    Args:
        inspectors: Inspector names to validate.
        registry: Set of known inspector names.

    Returns:
        The validated inspector names, deduplicated while preserving order.

    Raises:
        RecipeError: If any inspector name is not present in ``registry``.
    """
    seen: set[str] = set()
    validated: list[str] = []
    for name in inspectors:
        if name not in registry:
            msg = f"Unknown inspector: {name!r}"
            raise RecipeError(msg)
        if name not in seen:
            seen.add(name)
            validated.append(name)
    return tuple(validated)
