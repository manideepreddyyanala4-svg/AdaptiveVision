"""Unit tests for :mod:`adaptivevision.common`."""

from __future__ import annotations

import pytest

from adaptivevision import common as errors


def test_all_errors_derive_from_base() -> None:
    for cls in (
        errors.AcquisitionError,
        errors.CalibrationError,
        errors.InferenceError,
        errors.CommsError,
        errors.RecipeError,
        errors.FaultError,
    ):
        assert issubclass(cls, errors.AdaptiveVisionError)


def test_recoverable_defaults() -> None:
    assert errors.AcquisitionError("x").recoverable is True
    assert errors.InferenceError("x").recoverable is True
    assert errors.CommsError("x").recoverable is True
    assert errors.CalibrationError("x").recoverable is False
    assert errors.RecipeError("x").recoverable is False
    assert errors.FaultError("x").recoverable is False


def test_recoverable_override() -> None:
    err = errors.CalibrationError("x", recoverable=True)
    assert err.recoverable is True
    assert err.is_fatal is False


def test_is_fatal_is_negation_of_recoverable() -> None:
    assert errors.FaultError("x").is_fatal is True
    assert errors.AcquisitionError("x").is_fatal is False


def test_message_is_preserved() -> None:
    err = errors.CommsError("broker down")
    assert err.message == "broker down"
    assert str(err) == "broker down"


def test_can_be_raised_and_caught_as_base() -> None:
    with pytest.raises(errors.AdaptiveVisionError):
        raise errors.RecipeError("bad recipe")
