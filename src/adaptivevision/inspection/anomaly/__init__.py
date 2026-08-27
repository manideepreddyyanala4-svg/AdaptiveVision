"""AI-based anomaly detection inspectors (Milestone M9).

This package is part of the frozen structure defined in Architecture
Specification v1.0. It is intentionally empty at Milestone M0; its modules
are implemented in the milestone noted above.
"""

from adaptivevision.inspection.anomaly.detector import (
    StaticAnomalyDetector,
    ThresholdAnomalyDetector,
)

__all__ = ["StaticAnomalyDetector", "ThresholdAnomalyDetector"]
