"""Dimensional metrology inspection (Milestone M7)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from adaptivevision.alignment import LocalizedPart
from adaptivevision.common.enums import DefectClass, Severity
from adaptivevision.common.interfaces import Inspector
from adaptivevision.common.result import Defect, MetrologyResult
from adaptivevision.common.types import Measurement
from adaptivevision.recipe import Recipe

MeasurementSource = Callable[[LocalizedPart, Recipe], Mapping[str, float]]


class StaticMeasurementSource:
    """Deterministic measurement source for replay/tests/simulated stations."""

    def __init__(self, values: Mapping[str, float]) -> None:
        """Initialize with measured values keyed by measurement spec name."""
        self._values = dict(values)

    def measure(self, part: LocalizedPart, recipe: Recipe) -> Mapping[str, float]:
        """Return the configured measurements.

        Args:
            part: Localized part, accepted for interface compatibility.
            recipe: Active recipe, accepted for interface compatibility.
        """
        _ = (part, recipe)
        return dict(self._values)


class MetrologyInspector(Inspector[LocalizedPart, Recipe]):
    """Evaluate recipe measurement specs against measured feature values.

    Args:
        source: Callable that produces measured values in the units declared by
            each :class:`~adaptivevision.common.types.MeasurementSpec`.
    """

    def __init__(self, source: MeasurementSource) -> None:
        """Initialize the inspector."""
        self._source = source

    def inspect(self, part: LocalizedPart, recipe: Recipe) -> MetrologyResult:
        """Inspect an aligned part against recipe measurement specs."""
        values = self._source(part, recipe)
        measurements: list[Measurement] = []
        defects: list[Defect] = []

        for spec in recipe.measurement_specs:
            if spec.name not in values:
                defects.append(
                    Defect(
                        defect_class=DefectClass.DIMENSIONAL,
                        severity=Severity.CRITICAL,
                        description=f"Missing measurement for spec {spec.name!r}",
                    )
                )
                continue

            value = values[spec.name]
            in_tolerance = spec.contains(value)
            measurement = Measurement(
                name=spec.name,
                value=value,
                unit=spec.unit,
                spec=spec,
                in_tolerance=in_tolerance,
            )
            measurements.append(measurement)
            if not in_tolerance:
                defects.append(
                    Defect(
                        defect_class=DefectClass.DIMENSIONAL,
                        severity=Severity.MAJOR,
                        description=(
                            f"Measurement {spec.name!r}={value} {spec.unit} " f"outside tolerance"
                        ),
                    )
                )

        return MetrologyResult(measurements=tuple(measurements), defects=tuple(defects))
