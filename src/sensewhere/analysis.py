"""Aggregation of saved predictions into tables with subject-clustered bootstrap intervals.

The unit of resampling is the subject within a site, resampled independently per site; the reported number for a
method is the mean over sites of the mean over seeds, so every site counts equally regardless of its size.
"""
from __future__ import annotations

import numpy as np

from .metrics import balanced_accuracy, macro_f1


def _members(subjects: np.ndarray) -> list[np.ndarray]:
    uniq, inv = np.unique(subjects, return_inverse=True)
    return [np.flatnonzero(inv == k) for k in range(len(uniq))]


def multi_site_bootstrap(
    y: dict[str, np.ndarray],
    subjects: dict[str, np.ndarray],
    preds: dict[str, dict[str, np.ndarray]],
    n_classes: int,
    reference: str | None = None,
    metric=balanced_accuracy,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 95.0,
) -> dict[str, dict[str, float]]:
    """preds[site][method] has shape (n_seeds, n_windows). Returns per method: mean, lo, hi, and (if reference)
    diff, diff_lo, diff_hi against the reference method on identical resamples."""
    sites = sorted(y)
    methods = sorted({m for s in sites for m in preds[s]})
    members = {s: _members(subjects[s]) for s in sites}
    rng = np.random.default_rng(seed)

    def value(idx_by_site: dict[str, np.ndarray]) -> dict[str, float]:
        out = {}
        for m in methods:
            per_site = []
            for s in sites:
                if m not in preds[s]:
                    continue
                idx = idx_by_site[s]
                per_seed = [metric(y[s][idx], p[idx], n_classes) for p in preds[s][m]]
                per_site.append(np.nanmean(per_seed))
            out[m] = float(np.nanmean(per_site)) if per_site else float("nan")
        return out

    full = {s: np.arange(len(y[s])) for s in sites}
    point = value(full)
    draws = {m: np.empty(n_boot) for m in methods}
    for b in range(n_boot):
        idx = {}
        for s in sites:
            pick = rng.integers(0, len(members[s]), len(members[s]))
            idx[s] = np.concatenate([members[s][k] for k in pick])
        v = value(idx)
        for m in methods:
            draws[m][b] = v[m]
    res: dict[str, dict[str, float]] = {}
    for m in methods:
        a = (100.0 - ci) / 2
        lo, hi = np.nanpercentile(draws[m], [a, 100.0 - a])
        res[m] = {"mean": point[m], "lo": float(lo), "hi": float(hi)}
        if reference is not None and reference in draws:
            d = draws[m] - draws[reference]
            dlo, dhi = np.nanpercentile(d, [a, 100.0 - a])
            res[m].update({"diff": point[m] - point[reference], "diff_lo": float(dlo), "diff_hi": float(dhi)})
    return res


def per_site_scores(
    y: dict[str, np.ndarray], preds: dict[str, dict[str, np.ndarray]], n_classes: int, metric=balanced_accuracy
) -> dict[str, dict[str, float]]:
    """Mean over seeds of the metric on each site (no resampling)."""
    return {
        m: {s: float(np.nanmean([metric(y[s], p, n_classes) for p in preds[s][m]])) for s in sorted(y) if m in preds[s]}
        for m in sorted({m for s in y for m in preds[s]})
    }


__all__ = ["multi_site_bootstrap", "per_site_scores", "macro_f1", "balanced_accuracy"]
