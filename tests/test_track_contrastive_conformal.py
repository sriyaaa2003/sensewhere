import numpy as np
import pytest
import torch

from sensewhere import conformal, contrastive, models, track


# ------------------------------------------------------------------ track models
def _bags_data(n_tracks=24, per=7, seed=0):
    rng = np.random.default_rng(seed)
    X, y, keys, t = [], [], [], []
    for k in range(n_tracks):
        c = k % 3
        for w in range(per):
            x = rng.normal(0, 0.4, (16, 20)).astype(np.float32)
            x[2 + 4 * c : 5 + 4 * c, :] += 2.0
            X.append(x)
            y.append(c)
            keys.append(f"trk{k}")
            t.append(w * 0.25)
    return torch.from_numpy(np.stack(X)).unsqueeze(1), np.array(y), np.array(keys), np.array(t)


def test_index_tracks_orders_by_time_and_rejects_mixed_labels():
    X, y, keys, t = _bags_data(6, 4)
    perm = np.random.default_rng(1).permutation(len(y))
    ti = track.index_tracks(keys[perm], t[perm], y[perm])
    for idx in ti.members:
        assert (np.diff(t[perm][idx]) >= 0).all()
    assert len(ti.tracks) == 6 and ti.labels.shape == (6,)
    y_bad = y.copy()
    y_bad[0] = (y_bad[0] + 1) % 3
    with pytest.raises(ValueError):
        track.index_tracks(keys, t, y_bad)


def test_make_bags_mask_and_subsampling():
    X, y, keys, t = _bags_data(4, 6)
    ti = track.index_tracks(keys, t, y)
    bags, mask = track.make_bags(X, ti, np.arange(4), 8, None)
    assert bags.shape == (4, 8, 1, 16, 20) and mask.sum(1).tolist() == [6, 6, 6, 6]
    bags2, mask2 = track.make_bags(X, ti, np.arange(4), 3, np.random.default_rng(0))
    assert mask2.sum(1).tolist() == [3, 3, 3, 3]


@pytest.mark.parametrize("pool", ["attn", "gru"])
def test_track_model_forward_normalised_and_masked_windows_ignored(pool):
    torch.manual_seed(0)
    m = track.TrackModel(models.Backbone(widths=(4, 8), feat_dim=8), 3, pool=pool).eval()
    bags = torch.randn(2, 5, 1, 16, 20)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 3:] = False
    lp, w = m(bags, mask)
    assert torch.allclose(lp.exp().sum(1), torch.ones(2), atol=1e-5)
    bags2 = bags.clone()
    bags2[:, 3:] = torch.randn_like(bags2[:, 3:]) * 50  # change only the masked windows
    lp2, _ = m(bags2, mask)
    assert torch.allclose(lp, lp2, atol=1e-4)
    if pool == "attn":
        assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5) and (w[:, 3:] == 0).all()


@pytest.mark.parametrize("pool", ["attn", "gru"])
def test_fit_tracks_learns_and_predicts_whole_tracks(pool):
    X, y, keys, t = _bags_data(30, 6)
    ti = track.index_tracks(keys, t, y)
    Xv, yv, kv, tv = _bags_data(9, 6, seed=3)
    tiv = track.index_tracks(kv, tv, yv)
    m = track.TrackModel(models.Backbone(widths=(8, 16), feat_dim=16), 3, pool=pool)
    m, log = track.fit_tracks(m, X, ti, Xv, tiv, 3, epochs=30, tracks_per_batch=10, k=4, lr=3e-3, patience=30)
    lp, win = track.predict_tracks(m, Xv, tiv)
    assert (lp.argmax(1) == tiv.labels).mean() > 0.8
    assert win.shape == (len(yv),) and (win == yv).mean() > 0.8  # broadcast to every window of the track


# ------------------------------------------------------------------ contrastive
def test_nt_xent_is_low_for_matching_pairs_and_high_for_random():
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    assert contrastive.nt_xent(z, z + 0.01 * torch.randn_like(z)) < contrastive.nt_xent(z, torch.randn(32, 16))


def test_sample_pairs_use_two_distinct_windows_of_one_track():
    keys = np.array(["a"] * 4 + ["b"] * 3 + ["c"])  # c has one window and can never be sampled
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = contrastive.sample_pairs(keys, 2, rng)
        assert len(a) == 2 and (a != b).all()
        assert all(keys[i] == keys[j] and keys[i] != "c" for i, j in zip(a, b))


def test_contrastive_pretraining_lowers_the_loss():
    X, y, keys, t = _bags_data(40, 6)
    bb = models.Backbone(widths=(8, 16), feat_dim=16)
    losses = contrastive.pretrain_contrastive(bb, X, keys, steps=150, batch_tracks=16, lr=3e-3)
    assert losses[-1] < losses[0]


# ------------------------------------------------------------------ conformal
def test_conformal_quantile_gives_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    def draw(n):
        y = rng.integers(0, 3, n)
        logits = rng.normal(0, 1, (n, 3)); logits[np.arange(n), y] += 1.5
        p = np.exp(logits); return p / p.sum(1, keepdims=True), y
    pc, yc = draw(2000)
    pt, yt = draw(4000)
    q = conformal.calibrate(pc, yc, alpha=0.1)
    ev = conformal.evaluate_sets(conformal.prediction_sets(pt, q), yt)
    assert 0.87 < ev["coverage"] < 0.93


def test_prediction_sets_are_never_empty_and_larger_for_smaller_alpha():
    p = np.array([[0.4, 0.35, 0.25], [0.9, 0.05, 0.05]])
    for q in (0.0, 0.5, 1.0):
        s = conformal.prediction_sets(p, q)
        assert s.any(1).all()
    assert conformal.prediction_sets(p, 0.7).sum() >= conformal.prediction_sets(p, 0.2).sum()


def test_coverage_drops_under_shift_and_is_reported_honestly():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 2, n)
    easy = np.where(y[:, None] == np.arange(2), 0.9, 0.1)
    hard = np.where(y[:, None] == np.arange(2), 0.55, 0.45)
    q = conformal.calibrate(easy, y, 0.1)
    assert conformal.evaluate_sets(conformal.prediction_sets(easy, q), y)["coverage"] > 0.88
    shifted = hard.copy(); shifted[: n // 2] = shifted[: n // 2][:, ::-1]  # half the windows now point the wrong way
    assert conformal.evaluate_sets(conformal.prediction_sets(shifted, q), y)["coverage"] < 0.88


def test_sample_pairs_fails_loudly_when_no_track_has_two_windows():
    with pytest.raises(ValueError):
        contrastive.sample_pairs(np.array(["a", "b", "c"]), 2, np.random.default_rng(0))


def test_contrastive_reports_a_loss_even_for_fewer_than_fifty_steps():
    X, y, keys, t = _bags_data(20, 4)
    losses = contrastive.pretrain_contrastive(models.Backbone(widths=(4, 8), feat_dim=8), X, keys, steps=7, batch_tracks=8)
    assert len(losses) == 1 and np.isfinite(losses[0])
