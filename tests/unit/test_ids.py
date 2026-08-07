"""Unit tests for :mod:`adaptivevision.common.ids`."""

from __future__ import annotations

import re

from adaptivevision.common import ids

_PATTERN = re.compile(r"^(insp|part|frame|trace)-\d{13}-[0-9a-f]{8}$")


def test_all_generators_match_format() -> None:
    for value in (
        ids.new_inspection_id(),
        ids.new_part_id(),
        ids.new_frame_id(),
        ids.new_trace_id(),
    ):
        assert _PATTERN.match(value), value


def test_prefixes_are_correct() -> None:
    assert ids.new_inspection_id().startswith("insp-")
    assert ids.new_part_id().startswith("part-")
    assert ids.new_frame_id().startswith("frame-")
    assert ids.new_trace_id().startswith("trace-")


def test_ids_are_unique() -> None:
    generated = {ids.new_inspection_id() for _ in range(1000)}
    assert len(generated) == 1000


def test_injected_clock_and_random_are_deterministic() -> None:
    value = ids.new_inspection_id(now_ns=1_700_000_000_000_000_000, rand_hex="deadbeef")
    assert value == "insp-1700000000000-deadbeef"


def test_ids_are_time_ordered_by_millisecond() -> None:
    earlier = ids.new_part_id(now_ns=1_000_000_000_000_000, rand_hex="ffffffff")
    later = ids.new_part_id(now_ns=2_000_000_000_000_000, rand_hex="00000000")
    assert earlier < later
