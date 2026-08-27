"""Make ``benchmark.*`` importable when pytest is pointed at this directory directly."""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)
