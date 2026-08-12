"""Abstraction seams for AdaptiveVision.

These eight abstract base classes are the boundaries the domain and
orchestration layers depend on; concrete adapters are injected at the
composition root (Architecture Spec v1.0 §19). Per frozen decision 1 every seam
is an ABC (not a Protocol), giving explicit subclassing and the "cannot be
instantiated directly" guarantee that the Milestone M3 null-object strategy
relies on.

Two seams consume aggregates that are defined in later milestones - the aligned
part and the recipe. They are declared generic over :data:`PartT` and
:data:`RecipeT` so the contract is fixed here without inventing those types
(which belong to M6 and M2 respectively).

Errors referenced in ``Raises`` clauses are defined in
:mod:`adaptivevision.common.errors`. Storage failures from
:class:`ResultRepository` surface as the base
:class:`~adaptivevision.common.errors.AdaptiveVisionError`; a dedicated
``PersistenceError`` is a candidate change request for Milestone M4.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adaptivevision.common.result import (
        AnomalyResult,
        InspectionResult,
        PartialResult,
    )
    from adaptivevision.common.types import ROI, Image, RawFrame, RectifiedFrame

#: The aligned-part input to an inspector (concrete type defined in M6).
PartT = TypeVar("PartT")

#: The recipe aggregate (concrete type defined in M2).
RecipeT = TypeVar("RecipeT")


class CameraDriver(abc.ABC):
    """Seam for image acquisition devices (Milestone M3+).

    Implementations are not required to be thread-safe; a single acquisition
    thread owns the driver.
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Open the device and prepare it for capture.

        Raises:
            AcquisitionError: If the device cannot be opened.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device and its resources."""

    @abc.abstractmethod
    def capture(self, trigger_id: str | None = None) -> RawFrame:
        """Capture a single frame.

        Args:
            trigger_id: Identifier of the triggering event, if any.

        Returns:
            The acquired frame.

        Raises:
            AcquisitionError: On timeout, disconnect, or capture failure.
        """

    @abc.abstractmethod
    def is_healthy(self) -> bool:
        """Return ``True`` if the device is connected and operational."""


class InferenceEngine(abc.ABC):
    """Seam for a model inference backend (Milestone M8).

    Manages the lifecycle of a single loaded model.
    """

    @property
    @abc.abstractmethod
    def model_version(self) -> str:
        """Version identifier of the currently loaded model."""

    @abc.abstractmethod
    def load(self, model_id: str) -> None:
        """Load and prepare a model for inference.

        Args:
            model_id: Registry identifier of the model to load.

        Raises:
            InferenceError: If the model cannot be loaded.
        """

    @abc.abstractmethod
    def warmup(self) -> None:
        """Run warmup inferences to stabilize latency.

        Raises:
            InferenceError: If warmup fails.
        """

    @abc.abstractmethod
    def infer(self, inputs: Mapping[str, Image]) -> Mapping[str, Image]:
        """Run inference on named input tensors.

        Args:
            inputs: Mapping of input name to tensor.

        Returns:
            Mapping of output name to tensor.

        Raises:
            InferenceError: On execution failure.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Unload the model and free its resources."""


class AnomalyDetector(abc.ABC):
    """Seam for an anomaly-detection inspector backend (Milestone M9)."""

    @abc.abstractmethod
    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        """Score a frame (optionally restricted to a region) for anomalies.

        Args:
            frame: The rectified frame to analyze.
            roi: Optional region to restrict analysis to.

        Returns:
            The anomaly result including score and optional heatmap reference.

        Raises:
            InferenceError: On inference failure.
        """


class Inspector(abc.ABC, Generic[PartT, RecipeT]):
    """Seam for an inspection stage (Milestone M7+).

    Generic over the aligned-part input and the recipe, both defined in later
    milestones.
    """

    @abc.abstractmethod
    def inspect(self, part: PartT, recipe: RecipeT) -> PartialResult:
        """Inspect an aligned part against a recipe.

        Args:
            part: The localized, aligned part.
            recipe: The active recipe.

        Returns:
            The partial result contributed by this inspector.
        """


class PLCTransport(abc.ABC):
    """Seam for PLC register / coil transport over Modbus TCP (Milestone M11)."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the transport connection.

        Raises:
            CommsError: If the connection cannot be established.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the transport connection."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the transport is connected."""

    @abc.abstractmethod
    def read_coils(self, address: int, count: int) -> tuple[bool, ...]:
        """Read ``count`` coils starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """

    @abc.abstractmethod
    def write_coil(self, address: int, value: bool) -> None:
        """Write a single coil.

        Raises:
            CommsError: On communication failure.
        """

    @abc.abstractmethod
    def read_registers(self, address: int, count: int) -> tuple[int, ...]:
        """Read ``count`` holding registers starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """

    @abc.abstractmethod
    def write_registers(self, address: int, values: Sequence[int]) -> None:
        """Write holding registers starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """


class MessagePublisher(abc.ABC):
    """Seam for publishing messages to a broker (Milestone M12)."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the broker connection.

        Raises:
            CommsError: If the connection cannot be established.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the publisher is connected."""

    @abc.abstractmethod
    def publish(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        """Publish a payload to a topic.

        Args:
            topic: Destination topic.
            payload: JSON-serializable message body.
            qos: Delivery quality-of-service level.
            retain: Whether the broker should retain the message.

        Raises:
            CommsError: On publish failure.
        """


class ResultRepository(abc.ABC):
    """Seam for persisting and querying inspection results (Milestone M4)."""

    @abc.abstractmethod
    def save_result(self, result: InspectionResult) -> None:
        """Persist an inspection result.

        Raises:
            AdaptiveVisionError: On storage failure.
        """

    @abc.abstractmethod
    def get_result(self, inspection_id: str) -> InspectionResult | None:
        """Return the result with ``inspection_id``, or ``None`` if absent."""

    @abc.abstractmethod
    def list_results(self, *, limit: int = 100, offset: int = 0) -> tuple[InspectionResult, ...]:
        """Return a page of results ordered most-recent first."""


class RecipeStore(abc.ABC, Generic[RecipeT]):
    """Seam for recipe storage and versioning (Milestone M2).

    Generic over the recipe aggregate, defined in Milestone M2.
    """

    @abc.abstractmethod
    def load(self, recipe_id: str) -> RecipeT:
        """Load a recipe by identifier.

        Raises:
            RecipeError: If the recipe is missing or invalid.
        """

    @abc.abstractmethod
    def save(self, recipe: RecipeT) -> None:
        """Persist a recipe.

        Raises:
            RecipeError: On storage failure.
        """

    @abc.abstractmethod
    def list_ids(self) -> tuple[str, ...]:
        """Return the identifiers of all stored recipes."""
