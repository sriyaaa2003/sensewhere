import numpy as np
import pandas as pd
import pytest
import torch

from sensewhere import data, experiments as ex, models, splits, train

H, W = 16, 24
LEAVES = data.base_hierarchy().leaves


def _synthetic(n_per=14, seed=0):
    """4 sites x 12 atomic classes; the class is written into a row band so a CNN can learn it, sites add a gain offset."""
    rng = np.random.default_rng(seed)
    rows, X = [], []
    subj = 0
    for si, site in enumerate("ABCD"):
        for ci, atomic in enumerate(LEAVES):
            for k in range(n_per):
                if k % 7 == 0:
                    subj += 1
                x = rng.normal(0, 0.3, (H, W)).astype(np.float32)
                x[ci : ci + 2, :] += 2.0
                x += 0.3 * si
                X.append(x)
                rows.append(dict(task="motion", subject=subj, location=site, label_atomic=atomic, label_load=data.ATOMIC_TO_LOAD[atomic]))
    return np.stack(X), pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synth():
    X, meta = _synthetic()
    return X, meta


def test_encode_rejects_unknown_labels(synth):
    _, meta = synth
    bad = meta.copy()
    bad.loc[0, "label_atomic"] = "moonwalk"
    with pytest.raises(ValueError):
        ex.encode(bad, "atomic")


def test_hierarchy_check_passes_and_detects_corruption(synth):
    _, meta = synth
    data.check_hierarchy_consistency(meta)
    bad = meta.copy()
    bad.loc[0, "label_load"] = "On Wheels"
    with pytest.raises(ValueError):
        data.check_hierarchy_consistency(bad)


@pytest.mark.parametrize("head", ["flat", "hier", "moe"])
def test_cell_runs_for_every_head_on_atomic_level(synth, head):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[0]
    cell = ex.make_cell(X, meta, sp, "atomic")
    m, pred, log = ex.run_method(cell, "atomic", head, "erm", seed=0, epochs=6, batch_size=64, lr=3e-3)
    s = ex.score(cell.yte, pred, 12)
    assert s["bal_acc"] > 0.3 and s["n"] == len(sp.target_idx)


def test_adaptation_returns_all_methods_and_flags_collapse(synth):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[1]
    cell = ex.make_cell(X, meta, sp, "load")
    m, pred, _ = ex.run_method(cell, "load", "flat", "erm", seed=0, epochs=4, batch_size=64, lr=3e-3)
    res = ex.adapt_rows(m, cell, "load", tent_epochs=(1,), shot_epochs=(1,))
    assert set(res) == {"adabn", "tent_e1", "shot_e1"}
    for name, (p, tr) in res.items():
        assert len(p) == len(cell.yte) and 0.0 < ex.collapse_share(p, 3) <= 1.0
    assert ex.collapse_share(np.zeros(10, dtype=int), 3) == 1.0


@pytest.mark.parametrize("head", ["flat", "hier", "moe"])
def test_extension_keeps_old_classes_alive_and_learns_new(synth, head):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[2]
    cell = ex.make_cell(X, meta, sp, "atomic")
    base, _, _ = ex.run_method(cell, "atomic", head, "erm", seed=0, epochs=6, batch_size=64, lr=3e-3)
    rng = np.random.default_rng(0)
    # two new classes: a band at rows 14-15 and rows 12-13 that old classes do not use as strongly
    def newset(n, off):
        Xn = rng.normal(0, 0.3, (n, 1, H, W)).astype(np.float32)
        yn = rng.integers(0, 2, n)
        for i in range(n):
            Xn[i, 0, 13 + yn[i] : 15 + yn[i], :] += 2.0
        return torch.from_numpy(Xn), yn
    nx, ny = newset(80, 0)
    vx, vy = newset(30, 1)
    m, log = ex.extend_and_finetune(base, head, cell, nx, ny, vx, vy, 2, replay_frac=0.2, seed=0, epochs=5, names=["n0", "n1"])
    assert m.head.n_classes == 14
    pred, lp = train.predict(m, cell.Xte)
    assert lp.shape[1] == 14 and np.allclose(np.exp(lp).sum(1), 1, atol=1e-4)
    # the base model is untouched
    assert base.head.n_classes == 12


def test_explicit_two_level_hierarchy_at_load_level_and_extension():
    X, meta = _synthetic()
    h = data.Hierarchy({"root": ["normal_walking", "load", "On Wheels"], "load": ["Dual-hand Load"]},
                       ["normal_walking", "Dual-hand Load", "On Wheels"])
    sp = splits.leave_one_location_out(meta, seed=0)[0]
    cell = ex.make_cell(X, meta, sp, "load")
    m, pred, _ = ex.run_method(cell, "load", "hier", "erm", 0, 4, batch_size=64, lr=3e-3, hierarchy=h)
    assert m.head.n_classes == 3
    m.head.extend(1, parent="load", names=["Single-hand Load"])
    assert m.head.leaves == ["normal_walking", "Dual-hand Load", "On Wheels", "Single-hand Load"]
    lp = m(cell.Xte[:5])
    assert lp.shape == (5, 4) and torch.allclose(lp.exp().sum(1), torch.ones(5), atol=1e-4)


def test_inner_cells_never_contain_the_outer_target_and_are_subject_disjoint(synth):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[1]  # target B
    cells = ex.inner_cells(X, meta, sp, "load")
    assert len(cells) == 3
    for c in cells:
        assert "B" not in c.domain_sites and c.split.target != "B"
        assert len(c.yte) > 0 and len(c.ytr) > 0


def test_select_config_picks_the_better_learning_rate(synth):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[0]
    best, scores = ex.select_config(X, meta, sp, "load", "flat", "erm", [dict(lr=1e-7), dict(lr=3e-3)], seed=0, epochs=5, batch_size=64)
    assert best == 1 and scores[1] > scores[0]


def test_run_method_with_input_norm_profiles_and_transplant(synth):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[2]
    cell = ex.make_cell(X, meta, sp, "load")
    profs = {s: np.full(H, i, dtype=np.float32) for i, s in enumerate("ABCD")}
    m, pred, log = ex.run_method(cell, "load", "flat", "erm", 0, 3, batch_size=64, input_norm=True, profiles=profs, floor_transplant=True)
    assert m.input_norm and len(pred) == len(cell.yte)


def test_extension_with_frozen_backbone_leaves_backbone_and_its_batchnorm_statistics_untouched(synth):
    X, meta = synth
    sp = splits.leave_one_location_out(meta, seed=0)[3]
    cell = ex.make_cell(X, meta, sp, "atomic")
    base, _, _ = ex.run_method(cell, "atomic", "flat", "erm", seed=0, epochs=3, batch_size=64)
    rng = np.random.default_rng(1)
    nx = torch.from_numpy(rng.normal(5, 1, (60, 1, H, W)).astype(np.float32))
    ny = rng.integers(0, 2, 60)
    ref = {k: v.clone() for k, v in base.backbone.state_dict().items()}
    m, _ = ex.extend_and_finetune(base, "flat", cell, nx, ny, nx[:20], ny[:20], 2, replay_frac=0.0, seed=0, epochs=3)
    for k, v in m.backbone.state_dict().items():
        assert torch.equal(v, ref[k]), k
