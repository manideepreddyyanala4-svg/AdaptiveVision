"""Exception hierarchy for AdaptiveVision.

Every error derives from :class:`AdaptiveVisionError` and carries a
``recoverable`` flag. That flag is the contract consumed by the station's
failure-handling matrix and state machine (Architecture Spec v1.0 §17): a
recoverable error degrades a single part to ``REVIEW`` or triggers a retry,
while a non-recoverable error drives the station to a fault / safe state.

This module only *defines and raises* errors. It never catches, handles, or
logs them - those responsibilities belong to the orchestration and
infrastructure layers.
"""

from __future__ import annotations


class AdaptiveVisionError(Exception):
    """Base class for all AdaptiveVision errors.

    Attributes:
        message: Human-readable description of the error.
        recoverable: Whether the station can recover without intervention.
    """

    #: Default recoverability for the class, overridable per instance.
    default_recoverable: bool = False

    def __init__(self, message: str, *, recoverable: bool | None = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the error.
            recoverable: Explicit override of :attr:`default_recoverable`.
        """
        super().__init__(message)
        self.message = message
        self.recoverable = self.default_recoverable if recoverable is None else recoverable

    @property
    def is_fatal(self) -> bool:
        """Return ``True`` when the error is non-recoverable."""
        return not self.recoverable


class AcquisitionError(AdaptiveVisionError):
    """Image acquisition failure (camera timeout, disconnect, no frame)."""

    default_recoverable = True


class CalibrationError(AdaptiveVisionError):
    """Missing, invalid, or drifted calibration."""

    default_recoverable = False


class InferenceError(AdaptiveVisionError):
    """Inference engine load, warmup, or execution failure."""

    default_recoverable = True


class CommsError(AdaptiveVisionError):
    """Industrial communication failure (PLC / MQTT)."""

    default_recoverable = True


class RecipeError(AdaptiveVisionError):
    """Invalid, missing, or unloadable recipe."""

    default_recoverable = False


class FaultError(AdaptiveVisionError):
    """General station fault requiring intervention."""

    default_recoverable = False


class RetrievalError(AdaptiveVisionError):
    """Historical-defect vector retrieval failure (Milestone M19)."""

    default_recoverable = True


class AdvisoryError(AdaptiveVisionError):
    """Local LLM advisory (root-cause explanation) failure (Milestone M19)."""

    default_recoverable = True
