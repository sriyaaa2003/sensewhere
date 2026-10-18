"""Source-free recalibration of the walking vs dual-hand-load boundary at an unseen site.

The deployed classifier produces class probabilities for every window; tracks pool them. At a new site the boundary
between the two foot classes is often shifted (one site calls carriers "walking", another the reverse). These methods
move only that boundary, using nothing but the unlabeled target tracks' own scores: no labels, no gradient steps.

Class order is the repository's: 0 normal_walking, 1 dual-hand load, 2 on wheels.
"""
from __future__ import annotations

import numpy as np

WALK, DUAL, WHEELS = 0, 1, 2


def track_mean_probs(logp: np.ndarray, track_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean probability per track (T x C) and the index of each window's track."""
    p = np.exp(logp.astype(np.float64))
    uniq, inv = np.unique(track_ids, return_inverse=True)
    sums = np.zeros((len(uniq), p.shape[1]))
    np.add.at(sums, inv, p)
    return sums / np.bincount(inv)[:, None], inv


def foot_score(P: np.ndarray) -> np.ndarray:
    """Log-odds of dual-hand load against walking, per track."""
    return np.log(np.clip(P[:, DUAL], 1e-9, None)) - np.log(np.clip(P[:, WALK], 1e-9, None))


def is_foot(P: np.ndarray) -> np.ndarray:
    return P.argmax(1) != WHEELS


def decide(P: np.ndarray, bias: float = 0.0) -> np.ndarray:
    """Track labels: wheels when it is the most probable class, otherwise dual-hand iff score + bias > 0."""
    s = foot_score(P) + bias
    out = np.where(s > 0, DUAL, WALK)
    out[~is_foot(P)] = WHEELS
    return out


def bias_median(P: np.ndarray) -> float:
    """Put half of the foot tracks on each side of the boundary (assumes the two foot classes are about equally common)."""
    f = is_foot(P)
    return float(-np.median(foot_score(P)[f])) if f.sum() >= 4 else 0.0


def bias_mean(P: np.ndarray) -> float:
    f = is_foot(P)
    return float(-np.mean(foot_score(P)[f])) if f.sum() >= 4 else 0.0


def bias_gmm(P: np.ndarray, min_gap: float = 1.0) -> float:
    """Boundary at the equal-posterior point between the two components of a 1-D Gaussian mixture fitted to the foot scores.

    Falls back to no shift when the components are not clearly separated (distance between means below min_gap standard
    deviations) or when there are too few tracks."""
    from sklearn.mixture import GaussianMixture

    f = is_foot(P)
    s = foot_score(P)[f]
    if len(s) < 8:
        return 0.0
    gm = GaussianMixture(2, random_state=0, n_init=3).fit(s[:, None])
    mu = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    if abs(mu[0] - mu[1]) < min_gap * max(sd.mean(), 1e-6):
        return 0.0
    grid = np.linspace(mu.min(), mu.max(), 400)[:, None]
    post = gm.predict_proba(grid)
    hi = int(np.argmax(mu))
    k = np.argmin(np.abs(post[:, hi] - 0.5))
    return float(-grid[k, 0])


def em_priors(P: np.ndarray, train_prior: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    """Saerens et al. EM estimate of the target class priors from calibrated posteriors (train_prior defaults to uniform,
    which matches class-balanced training)."""
    c = P.shape[1]
    train_prior = np.full(c, 1.0 / c) if train_prior is None else train_prior
    pi = train_prior.copy()
    for _ in range(iters):
        adj = P * (pi / train_prior)
        adj /= adj.sum(1, keepdims=True)
        new = adj.mean(0)
        if np.abs(new - pi).max() < 1e-6:
            break
        pi = new
    return pi


def decide_em(P: np.ndarray) -> np.ndarray:
    pi = em_priors(P)
    adj = P * (pi / np.full(P.shape[1], 1.0 / P.shape[1]))
    return adj.argmax(1)


METHODS = {
    "none": lambda P: decide(P, 0.0),
    "em_prior": decide_em,
    "median": lambda P: decide(P, bias_median(P)),
    "mean": lambda P: decide(P, bias_mean(P)),
    "gmm": lambda P: decide(P, bias_gmm(P)),
}


def ensemble_logp(logps: list[np.ndarray]) -> np.ndarray:
    """Average of the members' log-probabilities, renormalised (geometric mean of the probabilities)."""
    m = np.mean(np.stack([lp.astype(np.float64) for lp in logps]), axis=0)
    m -= np.log(np.exp(m).sum(1, keepdims=True))
    return m
