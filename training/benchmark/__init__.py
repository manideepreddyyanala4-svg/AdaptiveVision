"""Model-zoo benchmark for the AdaptiveVision anomaly-detection stack.

The M9 training tree shipped two methods (a from-scratch autoencoder and
PaDiM) fitted on four hand-picked categories. This package generalizes that
into a sweep: every method in the zoo against every dataset configuration on
disk, scored with identical metrics so the results are directly comparable,
and ranked into a leaderboard.

Run everything from the repository root::

    python training/benchmark/run.py --models all --datasets all
    python training/benchmark/leaderboard.py

``training/`` is prepended to ``sys.path`` below so the benchmark modules can
reuse the existing flat helpers (``image_io``, ``datasets``, ``padim``)
regardless of whether the caller runs from the repo root or from ``training/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)
