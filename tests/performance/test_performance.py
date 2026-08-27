"""Performance smoke tests (Milestone M17).

These tests assert loose upper bounds on throughput so regressions in the hot
path are caught without being flaky on slow CI machines.
"""

from __future__ import annotations

import time

from adaptivevision.acquisition import NullCameraDriver
from adaptivevision.common.enums import CameraKind
from adaptivevision.config import CameraConfig
from adaptivevision.orchestration import InspectionPipeline, ResultBuffer


def test_pipeline_throughput_smoke() -> None:
    camera = NullCameraDriver(
        CameraConfig("cam0", CameraKind.AREA_SCAN_2D, 640, 480, 30.0)
    )
    camera.open()
    pipeline = InspectionPipeline(camera, station_id="station-1", recipe_ver="1.0.0")

    start = time.perf_counter()
    for i in range(50):
        pipeline.run(part_id=f"part-{i}")
    elapsed = time.perf_counter() - start
    # Loose bound: 50 cycles should complete well under 5 seconds.
    assert elapsed < 5.0


def test_buffer_throughput_smoke() -> None:
    buffer = ResultBuffer(capacity=100_000)
    start = time.perf_counter()
    for _i in range(10_000):
        buffer.push(object())  # type: ignore[arg-type]

    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
    assert len(buffer) == 10_000
