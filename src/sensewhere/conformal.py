"""Split-conformal prediction sets for track-level class probabilities, and a coverage check at an unseen site.

Calibration uses tracks from the SOURCE sites only. Under a site change the exchangeability assumption behind the
coverage guarantee fails, so the empirical coverage at the unseen site is what gets reported, not the nominal level.
"""
from __future__ import annotations

import numpy as np


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-ambiguous-classifier score: one minus the probability of the true class."""
    return 1.0 - probs[np.arange(len(y)), y]


def calibrate(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> float:
    """The finite-sample-corrected (1 - alpha) quantile of the calibration scores."""
    s = np.sort(lac_scores(probs_cal, y_cal))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1]) if n else 1.0


def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean (N, C): class c is in the set when 1 - p_c <= qhat. An empty set is replaced by the arg-max class."""
    sets = (1.0 - probs) <= qhat
    empty = ~sets.any(1)
    sets[empty, probs[empty].argmax(1)] = True
    return sets


def evaluate_sets(sets: np.ndarray, y: np.ndarray) -> dict:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    single = size == 1
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "singleton_share": float(single.mean()),
        "singleton_accuracy": float(covered[single].mean()) if single.any() else float("nan"),
    }
