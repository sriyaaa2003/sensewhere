import numpy as np
import pytest

from sensewhere import analysis


def _data(seed=0):
    rng = np.random.default_rng(seed)
    y, subj, preds = {}, {}, {}
    for s in "AB":
        n = 300
        y[s] = rng.integers(0, 3, n)
        subj[s] = np.repeat(np.arange(30), 10)
        good = np.where(rng.random(n) < 0.85, y[s], rng.integers(0, 3, n))
        bad = np.where(rng.random(n) < 0.45, y[s], rng.integers(0, 3, n))
        preds[s] = {"good": np.stack([good, good]), "bad": np.stack([bad, bad])}
    return y, subj, preds


def test_bootstrap_ranks_good_above_bad_and_diff_interval_excludes_zero():
    y, subj, preds = _data()
    r = analysis.multi_site_bootstrap(y, subj, preds, 3, reference="bad", n_boot=300)
    assert r["good"]["mean"] > r["bad"]["mean"]
    assert r["good"]["diff"] > 0 and r["good"]["diff_lo"] > 0
    assert r["bad"]["diff"] == 0 and r["bad"]["diff_lo"] == 0 and r["bad"]["diff_hi"] == 0
    for m in r.values():
        assert m["lo"] <= m["mean"] <= m["hi"] or m["hi"] - m["lo"] < 0.2


def test_bootstrap_is_reproducible_and_seed_sensitive():
    y, subj, preds = _data()
    a = analysis.multi_site_bootstrap(y, subj, preds, 3, n_boot=100, seed=1)
    b = analysis.multi_site_bootstrap(y, subj, preds, 3, n_boot=100, seed=1)
    c = analysis.multi_site_bootstrap(y, subj, preds, 3, n_boot=100, seed=2)
    assert a == b and a != c


def test_every_site_counts_equally():
    y = {"A": np.array([0, 1] * 5), "B": np.array([0, 1] * 500)}
    subj = {"A": np.arange(10), "B": np.repeat(np.arange(100), 10)}
    preds = {"A": {"m": np.stack([y["A"]])}, "B": {"m": np.stack([1 - y["B"]])}}
    r = analysis.multi_site_bootstrap(y, subj, preds, 2, n_boot=20)
    assert r["m"]["mean"] == pytest.approx(0.5)  # perfect on A, zero on B, equal weight


def test_per_site_scores_average_over_seeds():
    y = {"A": np.array([0, 1, 0, 1])}
    preds = {"A": {"m": np.stack([y["A"], 1 - y["A"]])}}
    assert analysis.per_site_scores(y, preds, 2)["m"]["A"] == pytest.approx(0.5)


def test_wider_confidence_level_gives_wider_interval():
    y, subj, preds = _data()
    a = analysis.multi_site_bootstrap(y, subj, preds, 3, reference="bad", n_boot=300, ci=95.0)
    b = analysis.multi_site_bootstrap(y, subj, preds, 3, reference="bad", n_boot=300, ci=99.4)
    assert b["good"]["diff_hi"] - b["good"]["diff_lo"] > a["good"]["diff_hi"] - a["good"]["diff_lo"]
    assert b["good"]["diff"] == a["good"]["diff"]
