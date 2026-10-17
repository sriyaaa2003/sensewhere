"""Metrics and cluster-aware statistics. Windows from one subject are not independent, so every interval resamples subjects."""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean recall over the classes present in y_true (absent classes are skipped, not counted as zero)."""
    cm = confusion(y_true, y_pred, n_classes)
    support = cm.sum(1)
    present = support > 0
    if not present.any():
        return float("nan")
    return float((np.diag(cm)[present] / support[present]).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 over the classes present in y_true or y_pred."""
    cm = confusion(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    present = (cm.sum(1) + cm.sum(0)) > 0
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[present].mean()) if present.any() else float("nan")


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = confusion(y_true, y_pred, n_classes)
    support = cm.sum(1)
    return np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)


def cluster_bootstrap(
    values_fn,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval of values_fn(index) with whole groups (subjects) resampled.

    values_fn receives an integer index array into the original rows.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    point = float(values_fn(np.arange(len(groups))))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        stats[b] = values_fn(idx)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_cluster_bootstrap(
    metric,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Interval for metric(A) - metric(B) on the same windows, resampling subjects."""

    def diff(idx: np.ndarray) -> float:
        return metric(y_true[idx], pred_a[idx], n_classes) - metric(y_true[idx], pred_b[idx], n_classes)

    return cluster_bootstrap(diff, groups, n_boot=n_boot, seed=seed)
