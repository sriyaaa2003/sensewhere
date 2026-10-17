import numpy as np
import pytest
import torch

from sensewhere import data, models, train
from sensewhere.metrics import balanced_accuracy

TREE = {"root": ["a", "b", "c"]}
LEAVES = ["a", "b", "c"]


def _toy(n=240, seed=0):
    """Three classes distinguished by the row where energy sits; two 'sites' with different gain."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    dom = rng.integers(0, 2, n)
    X = rng.normal(0, 0.3, (n, 1, 16, 16)).astype(np.float32)
    for i in range(n):
        X[i, 0, 3 + 4 * y[i] : 6 + 4 * y[i], :] += 2.0 * (1.0 + 0.5 * dom[i])
    return torch.from_numpy(X), y, dom


def _model(kind="flat"):
    torch.manual_seed(0)
    return models.Classifier(models.Backbone(widths=(8, 16), feat_dim=16), models.make_head(kind, 16, 3, tree=TREE, leaves=LEAVES, n_experts=2, top_k=2))


@pytest.mark.parametrize("method,augment", [("erm", False), ("erm", True), ("coral", False), ("dann", False), ("groupdro", False)])
def test_every_method_trains_and_beats_chance(method, augment):
    X, y, d = _toy()
    Xv, yv, _ = _toy(90, seed=1)
    cfg = train.TrainConfig(method=method, augment=augment, epochs=8, batch_size=32, lr=3e-3, patience=8)
    m, log = train.fit(_model(), X, y, d, Xv, yv, 3, cfg)
    p, _ = train.predict(m, Xv)
    assert balanced_accuracy(yv, p, 3) > 0.6
    assert all(np.isfinite(r["loss"]) for r in log)


@pytest.mark.parametrize("kind", ["flat", "hier", "moe"])
def test_every_head_trains(kind):
    X, y, d = _toy()
    Xv, yv, _ = _toy(90, seed=1)
    cfg = train.TrainConfig(epochs=8, batch_size=32, lr=3e-3, patience=8)
    m, _ = train.fit(_model(kind), X, y, d, Xv, yv, 3, cfg)
    p, _ = train.predict(m, Xv)
    assert balanced_accuracy(yv, p, 3) > 0.6


def test_best_epoch_is_restored_by_source_validation():
    X, y, d = _toy()
    Xv, yv, _ = _toy(90, seed=1)
    cfg = train.TrainConfig(epochs=6, batch_size=32, lr=3e-3, patience=6)
    m, log = train.fit(_model(), X, y, d, Xv, yv, 3, cfg)
    p, _ = train.predict(m, Xv)
    assert balanced_accuracy(yv, p, 3) == pytest.approx(max(r["val_bal_acc"] for r in log), abs=1e-6)


def test_same_seed_is_reproducible_on_cpu():
    X, y, d = _toy()
    Xv, yv, _ = _toy(90, seed=1)
    cfg = train.TrainConfig(epochs=3, batch_size=32, seed=3)
    a, _ = train.fit(_model(), X, y, d, Xv, yv, 3, cfg)
    b, _ = train.fit(_model(), X, y, d, Xv, yv, 3, cfg)
    pa, la = train.predict(a, Xv)
    pb, lb = train.predict(b, Xv)
    assert np.allclose(la, lb, atol=1e-5)


def test_class_weights_balance_counts():
    w = train.class_weights(np.array([0] * 90 + [1] * 10), 2).numpy()
    assert w[1] / w[0] == pytest.approx(9.0)
    assert train.class_weights(np.array([0, 0]), 3).numpy()[2] == 0


def test_hierarchy_roll_up_is_consistent():
    h = data.base_hierarchy()
    assert set(h.leaves) == set(data.ATOMIC_TO_LOAD)
    assert set(data.ATOMIC_TO_LOAD.values()) == set(data.LOAD_CLASSES)
    head = models.HierHead(8, h.tree, h.leaves)
    assert head.n_classes == 12


def test_augmentation_is_off_unless_requested():
    """Regression: v1 silently augmented every non-ERM method, confounding method with augmentation."""
    X, y, d = _toy(64)
    Xv, yv, _ = _toy(30, seed=1)
    seen = []
    orig = train.augment
    train.augment = lambda xb, cfg, g: (seen.append(1), orig(xb, cfg, g))[1]
    try:
        for method in ["erm", "coral", "dann", "groupdro"]:
            train.fit(_model(), X, y, d, Xv, yv, 3, train.TrainConfig(method=method, epochs=1, batch_size=32), n_domains=2)
        assert not seen
        train.fit(_model(), X, y, d, Xv, yv, 3, train.TrainConfig(method="erm", augment=True, epochs=1, batch_size=32), n_domains=2)
        assert seen
        seen.clear()
        train.fit(_model(), X, y, d, Xv, yv, 3, train.TrainConfig(method="coral", legacy_aug=True, epochs=1, batch_size=32), n_domains=2)
        assert seen
    finally:
        train.augment = orig


def test_freeze_bn_keeps_running_statistics_fixed_during_finetuning():
    """Regression: a frozen backbone still updated BatchNorm running stats, silently changing old-class features."""
    X, y, d = _toy(64)
    Xv, yv, _ = _toy(30, seed=1)
    m = _model()
    for p in m.backbone.parameters():
        p.requires_grad_(False)
    bn = [x for x in m.backbone.modules() if isinstance(x, torch.nn.BatchNorm2d)][0]
    before = bn.running_mean.clone()
    train.fit(m, X + 5.0, y, d, Xv, yv, 3, train.TrainConfig(epochs=2, batch_size=32, freeze_bn=True), n_domains=2)
    assert torch.equal(bn.running_mean, before)
    m2 = _model()
    for p in m2.backbone.parameters():
        p.requires_grad_(False)
    bn2 = [x for x in m2.backbone.modules() if isinstance(x, torch.nn.BatchNorm2d)][0]
    b2 = bn2.running_mean.clone()
    train.fit(m2, X + 5.0, y, d, Xv, yv, 3, train.TrainConfig(epochs=2, batch_size=32, freeze_bn=False), n_domains=2)
    assert not torch.equal(bn2.running_mean, b2)


def test_floor_transplant_runs_and_uses_profiles():
    X, y, d = _toy(64)
    Xv, yv, _ = _toy(30, seed=1)
    prof = torch.randn(2, 16)
    cfg = train.TrainConfig(epochs=2, batch_size=32, floor_transplant=True, floor_profiles=prof)
    m, log = train.fit(_model(), X, y, d, Xv, yv, 3, cfg, n_domains=2)
    assert all(np.isfinite(r["loss"]) for r in log)


def test_input_norm_makes_classifier_invariant_to_offset_and_gain():
    m = models.Classifier(models.Backbone(widths=(4, 8), feat_dim=8), models.FlatHead(8, 3), input_norm=True).eval()
    x = torch.randn(4, 1, 16, 16)
    assert torch.allclose(m(x), m(3.0 * x + 7.0), atol=1e-4)


def test_fit_survives_all_nan_validation():
    X, y, d = _toy(64)
    Xv = torch.full((8, 1, 16, 16), float("nan"))
    m, log = train.fit(_model(), X, y, d, Xv, np.zeros(8, dtype=int), 3, train.TrainConfig(epochs=2, batch_size=32), n_domains=2)
    assert m is not None


def test_fit_with_fewer_windows_than_a_batch_still_takes_steps():
    X, y, d = _toy(10)
    X, y, d = X[:20], y[:20], d[:20]
    Xv, yv, _ = _toy(30, seed=1)
    m, log = train.fit(_model(), X, y, d, Xv, yv, 3, train.TrainConfig(epochs=3, batch_size=256), n_domains=2)
    assert log[0]["loss"] > 0
