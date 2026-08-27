"""A tiny rank-based AUROC, so training doesn't need a scikit-learn dependency."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Args:
        scores: Anomaly scores, higher means more anomalous.
        labels: Boolean (or 0/1) ground-truth anomaly labels.

    Returns:
        AUROC in ``[0, 1]``, or ``nan`` if only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
