import numpy as np
import pandas as pd
import pytest
import torch

from sensewhere import adapt, dg, metrics, models, splits

TREE = {"root": ["walk", "load", "wheels"], "load": ["pockets", "carry"]}
LEAVES = ["walk", "pockets", "carry", "wheels"]


# ---------------- metrics
def test_balanced_accuracy_skips_absent_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    assert metrics.balanced_accuracy(y, p, 4) == pytest.approx(0.75)


def test_macro_f1_perfect_and_worst():
    y = np.array([0, 1, 2, 2])
    assert metrics.macro_f1(y, y, 3) == pytest.approx(1.0)
    assert metrics.macro_f1(y, (y + 1) % 3, 3) == pytest.approx(0.0)


def test_cluster_bootstrap_resamples_groups_and_covers_point():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 5)
    y = rng.integers(0, 2, len(groups))
    pred = np.where(rng.random(len(groups)) < 0.8, y, 1 - y)
    point, lo, hi = metrics.cluster_bootstrap(lambda i: metrics.balanced_accuracy(y[i], pred[i], 2), groups, n_boot=300)
    assert lo <= point <= hi
    assert hi - lo < 0.5


def test_paired_bootstrap_identical_predictions_gives_zero():
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(10), 4)
    d, lo, hi = metrics.paired_cluster_bootstrap(metrics.macro_f1, y, y, y, groups, 2, n_boot=100)
    assert d == 0 and lo == 0 and hi == 0


# ---------------- splits
def _meta():
    rows = []
    for loc_i, loc in enumerate("ABCD"):
        for s in range(10):
            for _ in range(6):
                rows.append({"location": loc, "subject": f"{loc}{s}"})
    return pd.DataFrame(rows)


def test_lolo_splits_are_subject_disjoint_and_cover_everything():
    meta = _meta()
    out = splits.leave_one_location_out(meta, seed=1)
    assert [s.target for s in out] == list("ABCD")
    for s in out:
        splits.assert_disjoint(s, meta)
        assert set(meta.iloc[s.target_idx].location) == {s.target}
        assert s.target not in set(meta.iloc[s.train_idx].location) | set(meta.iloc[s.val_idx].location)
        assert len(s.train_idx) + len(s.val_idx) + len(s.target_idx) == len(meta)
        assert len(s.val_idx) > 0


def test_lolo_rejects_subject_in_two_locations():
    meta = _meta()
    meta.loc[0, "subject"] = "B0"
    with pytest.raises(ValueError):
        splits.leave_one_location_out(meta)


# ---------------- heads
def test_all_heads_return_normalised_log_probs():
    f = torch.randn(7, 16)
    for head in [
        models.FlatHead(16, 4),
        models.HierHead(16, TREE, LEAVES),
        models.MoEHead(16, 4, n_experts=3, top_k=2),
    ]:
        lp = head(f)
        assert lp.shape == (7, 4)
        assert torch.allclose(lp.exp().sum(1), torch.ones(7), atol=1e-5)


def test_hier_rejects_inconsistent_leaves():
    with pytest.raises(ValueError):
        models.HierHead(8, TREE, ["walk", "pockets", "wheels"])


def test_flat_extend_keeps_old_logits_close_and_adds_class():
    head = models.FlatHead(8, 3)
    f = torch.randn(5, 8)
    before = head.fc(f).detach()
    new_ids = head.extend(2)
    after = head.fc(f).detach()
    assert new_ids == [3, 4] and head.n_classes == 5
    assert torch.allclose(before, after[:, :3])


def test_hier_extend_only_changes_parent_node():
    head = models.HierHead(8, TREE, LEAVES)
    other = {k: v.weight.clone() for k, v in head.nodes.items() if k != "load"}
    head.extend(1, parent="load", names=["fold"])
    assert head.n_classes == 5 and head.leaves[-1] == "fold"
    for k, w in other.items():
        assert torch.equal(head.nodes[k].weight, w)
    lp = head(torch.randn(4, 8))
    assert lp.shape == (4, 5) and torch.allclose(lp.exp().sum(1), torch.ones(4), atol=1e-5)


def test_moe_extend_adds_expert_and_freezes_old():
    head = models.MoEHead(8, 3, n_experts=2, top_k=2)
    head.extend(1, add_expert=True, freeze_old=True)
    assert head.n_classes == 4 and len(head.experts) == 3
    assert not any(p.requires_grad for p in head.experts[0].parameters())
    assert all(p.requires_grad for p in head.experts[2].parameters())
    lp = head(torch.randn(6, 8))
    assert lp.shape == (6, 4) and torch.isfinite(lp).all()


def test_moe_balanced_gate_has_aux_loss_near_one():
    head = models.MoEHead(8, 3, n_experts=4, top_k=4)
    torch.nn.init.zeros_(head.gate.weight)
    torch.nn.init.zeros_(head.gate.bias)
    head(torch.randn(32, 8))
    assert head.aux_loss().item() == pytest.approx(1.0, abs=1e-5)


# ---------------- domain generalisation
def test_coral_zero_for_identical_domains_and_positive_otherwise():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    d = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
    same = torch.cat([x[:32], x[:32]])
    assert dg.coral_loss(same, d).item() == pytest.approx(0.0, abs=1e-6)
    scaled = same.clone()
    scaled[d == 1] *= 3
    assert dg.coral_loss(scaled, d).item() > 0.1


def test_coral_single_domain_is_zero():
    assert dg.coral_loss(torch.randn(10, 4), torch.zeros(10, dtype=torch.long)).item() == 0.0


def test_grad_reverse_flips_gradient_sign():
    x = torch.ones(3, requires_grad=True)
    dg.grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_group_dro_upweights_worst_domain():
    dro = dg.GroupDRO(2, step=1.0)
    loss = torch.tensor([0.1, 0.1, 2.0, 2.0])
    dom = torch.tensor([0, 0, 1, 1])
    for _ in range(5):
        dro.weighted(loss, dom)
    assert dro.q[1] > dro.q[0]
    assert dro.q.sum().item() == pytest.approx(1.0)


# ---------------- adaptation
def _toy_model(kind="flat"):
    torch.manual_seed(0)
    bb = models.Backbone(widths=(4, 8), feat_dim=8)
    head = models.make_head(kind, 8, 4, tree=TREE, leaves=LEAVES, n_experts=2, top_k=2)
    return models.Classifier(bb, head).eval()


def _batches(shift=0.0, n=4):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(8, 1, 16, 16, generator=g) * (1 + shift) + shift for _ in range(n)]


def test_adabn_updates_running_stats_without_changing_weights():
    m = _toy_model()
    w0 = m.head.fc.weight.clone()
    bn = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)][0]
    before = bn.running_mean.clone()
    out, trace = adapt.adabn(m, _batches(shift=2.0))
    bn2 = [x for x in out.modules() if isinstance(x, torch.nn.BatchNorm2d)][0]
    assert not torch.allclose(bn2.running_mean, before)
    assert torch.equal(out.head.fc.weight, w0)
    assert torch.equal(bn.running_mean, before)  # original untouched
    assert len(trace) == 4


@pytest.mark.parametrize("kind", ["flat", "hier", "moe"])
def test_tent_lowers_entropy_and_only_moves_bn_affine(kind):
    m = _toy_model(kind)
    frozen = {n: p.clone() for n, p in m.named_parameters() if "bn" not in n.lower() and "blocks.1." not in n and "blocks.5." not in n}
    out, trace = adapt.tent(m, _batches(), lr=5e-2, epochs=6)
    assert trace[-1] <= trace[0] + 1e-6
    for n, p in out.named_parameters():
        if n in frozen and p.dim() > 1:
            assert torch.equal(p, frozen[n]), n


def test_shot_im_keeps_head_frozen_and_changes_backbone():
    m = _toy_model()
    head_w = m.head.fc.weight.clone()
    bb_w = m.backbone.proj.weight.clone()
    out, trace = adapt.shot_im(m, _batches(), lr=1e-2, epochs=2)
    assert torch.equal(out.head.fc.weight, head_w)
    assert not torch.equal(out.backbone.proj.weight, bb_w)
    assert len(trace) == 8


def test_adaptation_does_not_mutate_input_model():
    m = _toy_model()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    adapt.tent(m, _batches(), epochs=1)
    adapt.shot_im(m, _batches(), epochs=1)
    adapt.adabn(m, _batches())
    for k, v in m.state_dict().items():
        assert torch.equal(v, sd[k]), k


def test_shared_group_split_keeps_validation_groups_out_of_training():
    rows = [{"loc": l, "user": u} for l in range(5) for u in range(9) for _ in range(4)]
    meta = pd.DataFrame(rows)
    out = splits.leave_one_domain_out_shared(meta, "loc", "user", val_fraction=0.2, seed=3)
    assert len(out) == 5
    for s in out:
        tr = set(meta.iloc[s.train_idx].user)
        va = set(meta.iloc[s.val_idx].user)
        assert not tr & va and va
        assert {str(v) for v in meta.iloc[s.target_idx]["loc"]} == {s.target}
        assert s.target not in set(meta.iloc[s.train_idx]["loc"]) | set(meta.iloc[s.val_idx]["loc"])
        assert len(s.train_idx) + len(s.val_idx) + len(s.target_idx) == len(meta)
