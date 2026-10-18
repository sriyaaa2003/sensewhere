import numpy as np
import pytest

from sensewhere import recal


def _scores_to_P(s, wheels=None):
    """Build 3-class probabilities whose foot log-odds equal s (a wheels track gets probability 0.9 on wheels)."""
    s = np.asarray(s, dtype=float)
    pw = 1.0 / (1.0 + np.exp(s))
    pd_ = 1.0 - pw
    P = np.stack([pw * 0.97, pd_ * 0.97, np.full(len(s), 0.03)], axis=1)
    if wheels is not None:
        P[wheels] = [0.05, 0.05, 0.9]
    return P / P.sum(1, keepdims=True)


def test_track_mean_probs_and_window_index():
    lp = np.log(np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]))
    P, inv = recal.track_mean_probs(lp, np.array(["a", "a", "b"]))
    assert P.shape == (2, 3) and inv.tolist() == [0, 0, 1]
    assert P[0, 0] == pytest.approx(0.7)


def test_foot_score_sign_and_decide_with_wheels():
    P = _scores_to_P([-2.0, 2.0, 0.5], wheels=[2])
    assert recal.decide(P).tolist() == [recal.WALK, recal.DUAL, recal.WHEELS]
    assert recal.decide(P, bias=-3.0)[1] == recal.WALK  # a negative bias moves the boundary toward walking


def test_bias_median_recentres_a_shifted_site():
    rng = np.random.default_rng(0)
    true = np.array([0] * 40 + [1] * 40)
    s = np.where(true == 0, rng.normal(-1.5, 0.7, 80), rng.normal(1.5, 0.7, 80)) + 3.0  # whole site shifted toward "dual"
    P = _scores_to_P(s)
    before = (recal.decide(P) == true).mean()
    after = (recal.decide(P, recal.bias_median(P)) == true).mean()
    assert after > before + 0.2 and after > 0.9


def test_gmm_boundary_recovers_shift_and_is_inactive_without_separation():
    rng = np.random.default_rng(1)
    true = np.array([0] * 60 + [1] * 60)
    s = np.where(true == 0, rng.normal(-2, 0.6, 120), rng.normal(2, 0.6, 120)) - 2.5  # shifted toward "walking"
    P = _scores_to_P(s)
    after = (recal.decide(P, recal.bias_gmm(P)) == true).mean()
    assert after > 0.92
    blob = _scores_to_P(rng.normal(0, 1, 100))
    assert recal.bias_gmm(blob) == 0.0  # one blob: no clear boundary, so no move


def test_gmm_with_few_tracks_does_nothing():
    assert recal.bias_gmm(_scores_to_P([1.0, -1.0, 2.0])) == 0.0
    assert recal.bias_median(_scores_to_P([1.0, -1.0])) == 0.0


def test_em_priors_find_an_imbalanced_target():
    rng = np.random.default_rng(2)
    y = rng.choice(3, size=3000, p=[0.15, 0.7, 0.15])
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), y] += 2.5
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    pi = recal.em_priors(P)
    assert abs(pi[1] - 0.7) < 0.08 and abs(pi.sum() - 1) < 1e-9


def test_ensemble_logp_is_normalised_geometric_mean():
    a = np.log(np.array([[0.7, 0.2, 0.1]]))
    b = np.log(np.array([[0.5, 0.4, 0.1]]))
    e = recal.ensemble_logp([a, b])
    assert np.exp(e).sum() == pytest.approx(1.0)
    assert e[0].argmax() == 0


def test_every_method_returns_valid_labels():
    P = _scores_to_P(np.random.default_rng(3).normal(0, 2, 50), wheels=[0, 1, 2])
    for name, fn in recal.METHODS.items():
        out = fn(P)
        assert out.shape == (50,) and set(out.tolist()) <= {0, 1, 2}, name
