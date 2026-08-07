"""Unit tests for :mod:`adaptivevision.common.timing`."""

from __future__ import annotations

from collections.abc import Callable

from adaptivevision.common import timing


def make_clock(times: list[float]) -> Callable[[], float]:
    """Return a clock that yields successive values from ``times``."""
    iterator = iter(times)
    return lambda: next(iterator)


def test_stopwatch_elapsed() -> None:
    clock = make_clock([100.0, 100.25])
    watch = timing.Stopwatch(clock=clock)
    assert watch.elapsed_s() == 0.25


def test_stopwatch_elapsed_ms() -> None:
    clock = make_clock([10.0, 10.5])
    watch = timing.Stopwatch(clock=clock)
    assert watch.elapsed_ms() == 500.0


def test_stopwatch_reset() -> None:
    clock = make_clock([1.0, 5.0, 6.0])
    watch = timing.Stopwatch(clock=clock)  # start = 1.0
    watch.reset()  # start = 5.0
    assert watch.elapsed_s() == 1.0  # 6.0 - 5.0


def test_deadline_not_expired_then_expired() -> None:
    clock = make_clock([0.0, 0.4, 1.1])
    deadline = timing.Deadline(1.0, clock=clock)  # deadline = 1.0
    assert deadline.expired() is False  # now 0.4
    assert deadline.expired() is True  # now 1.1


def test_deadline_remaining() -> None:
    clock = make_clock([0.0, 0.3])
    deadline = timing.Deadline(1.0, clock=clock)
    assert deadline.remaining_s() == 0.7


def test_deadline_from_ms() -> None:
    clock = make_clock([0.0, 0.25])
    deadline = timing.Deadline.from_ms(500.0, clock=clock)  # deadline = 0.5
    assert deadline.expired() is False


def test_measure_context_manager() -> None:
    clock = make_clock([2.0, 2.75])
    with timing.measure(clock=clock) as watch:
        pass
    assert watch.elapsed_ms() == 750.0
