import numpy as np
import pytest
import torch

from sensewhere import adapt, models, train


def test_track_vote_gives_each_track_one_label_and_follows_the_majority():
    # track 0: three windows favour class 1, one outlier favours class 0; track 1: all favour class 2
    p = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.3, 0.7, 0], [0.9, 0.1, 0], [0, 0.1, 0.9], [0, 0.2, 0.8]])
    lab, conf = adapt.track_vote(np.log(p + 1e-9), np.array([0, 0, 0, 0, 1, 1]))
    assert lab.tolist() == [1, 1, 1, 1, 2, 2]
    assert conf[0] == conf[3] and conf[4] == conf[5]


def _toy(n_tracks=24, per=8, seed=0):
    """Three classes by a row band; per-window noise; each track holds one class."""
    rng = np.random.default_rng(seed)
    X, y, tr = [], [], []
    for t in range(n_tracks):
        c = t % 3
        for _ in range(per):
            x = rng.normal(0, 0.4, (16, 16)).astype(np.float32)
            x[2 + 4 * c : 5 + 4 * c, :] += 2.0
            X.append(x)
            y.append(c)
            tr.append(t)
    return torch.from_numpy(np.stack(X)).unsqueeze(1), np.array(y), np.array(tr)


def _trained(seed=0):
    X, y, _ = _toy(seed=seed)
    Xv, yv, _ = _toy(6, seed=seed + 5)
    m = models.Classifier(models.Backbone(widths=(8, 16), feat_dim=16), models.FlatHead(16, 3))
    cfg = train.TrainConfig(epochs=10, batch_size=32, lr=3e-3, patience=10)
    m, _ = train.fit(m, X, y, np.zeros(len(y), dtype=int), Xv, yv, 3, cfg, n_domains=1)
    return m


def test_tcst_keeps_source_model_and_only_moves_bn_scale_shift():
    m = _trained()
    Xt, yt, tr = _toy(seed=11)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    out, trace = adapt.tcst(m, Xt + 0.5, tr, rounds=2)
    for k, v in m.state_dict().items():
        assert torch.equal(v, sd[k]), k
    base = dict(m.named_parameters())
    changed = [n for n, p in out.named_parameters() if not torch.equal(p, base[n])]
    assert changed and all(("blocks.1" in n or "blocks.5" in n) for n in changed)
    assert len(trace) > 0


def test_tcst_does_not_break_a_good_model_and_is_deterministic():
    m = _trained()
    Xt, yt, tr = _toy(seed=11)
    a, _ = adapt.tcst(m, Xt, tr, rounds=2)
    b, _ = adapt.tcst(m, Xt, tr, rounds=2)
    pa, _ = train.predict(a, Xt)
    pb, _ = train.predict(b, Xt)
    assert (pa == pb).all()
    assert (pa == yt).mean() > 0.7


def test_shot_fixed_variants_run_and_head_stays_frozen():
    m = _trained()
    Xt, yt, _ = _toy(seed=12)
    hw = m.head.fc.weight.clone()
    for params in ["bn", "all"]:
        out, trace = adapt.shot(m, Xt, epochs=2, params=params)
        assert torch.equal(out.head.fc.weight, hw) and len(trace) > 0
    with pytest.raises(ValueError):
        adapt.shot(m, Xt, params="nope")
