import numpy as np
import pytest
import torch

from sensewhere import mae as ssl, models, proto


# ------------------------------------------------------------------ prototypes
def _emb_model():
    torch.manual_seed(0)
    return models.Classifier(models.Backbone(widths=(8, 16), feat_dim=16), models.FlatHead(16, 3)).eval()


def test_class_means_are_unit_norm_and_absent_classes_are_zero():
    emb = torch.nn.functional.normalize(torch.randn(20, 8), dim=1)
    y = np.array([0] * 10 + [2] * 10)
    m = proto.class_means(emb, y, 4)
    assert m.shape == (4, 8) and m[1].abs().sum() == 0 and m[3].abs().sum() == 0
    assert torch.allclose(m[0].norm(), torch.tensor(1.0), atol=1e-5)


def test_ncm_classifies_separated_clusters_and_extension_keeps_old_prototypes():
    g = torch.Generator().manual_seed(0)
    centres = torch.nn.functional.normalize(torch.randn(3, 8, generator=g), dim=1)
    def sample(c, n):
        return torch.nn.functional.normalize(centres[c] + 0.1 * torch.randn(n, 8, generator=g), dim=1)
    means = proto.class_means(torch.cat([sample(0, 30), sample(1, 30)]), np.array([0] * 30 + [1] * 30), 2)
    old = means.clone()
    pred = proto.ncm_predict(torch.cat([sample(0, 10), sample(1, 10)]), means)
    assert (pred == np.array([0] * 10 + [1] * 10)).all()
    ext = proto.extend_means(means, sample(2, 5))
    assert ext.shape == (3, 8) and torch.equal(ext[:2], old)
    assert (proto.ncm_predict(sample(2, 10), ext) == 2).all()
    assert (proto.ncm_predict(sample(0, 10), ext) == 0).all()  # old class survives without any retraining


def test_embed_returns_unit_vectors():
    m = _emb_model()
    e = proto.embed(m, torch.randn(10, 1, 16, 16))
    assert e.shape == (10, 16) and torch.allclose(e.norm(dim=1), torch.ones(10), atol=1e-4)


# ------------------------------------------------------------------ self-supervised pretraining
def test_patch_mask_ratio_and_shape():
    g = torch.Generator().manual_seed(0)
    m = ssl.random_patch_mask((64, 1, 64, 90), (8, 9), 0.5, g)
    assert m.shape == (64, 1, 64, 90) and m.dtype == torch.bool
    assert 0.4 < m.float().mean() < 0.6


def test_masked_ae_reconstructs_to_input_shape_for_real_and_toy_sizes():
    for shape in [(2, 1, 64, 90), (2, 1, 16, 24)]:
        bb = models.Backbone(widths=(8, 16, 16, 16), feat_dim=8)
        out = ssl.MaskedAE(bb)(torch.randn(*shape))
        assert out.shape == shape


def test_pretraining_lowers_reconstruction_loss_on_structured_windows():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6.28, 24)[None, None, :]
    f = np.linspace(0, 3.14, 16)[None, :, None]
    X = np.sin(f * rng.uniform(1, 3, (256, 1, 1)) + t) + 0.05 * rng.normal(size=(256, 16, 24))
    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
    bb = models.Backbone(widths=(8, 16), feat_dim=8)
    losses = ssl.pretrain(bb, X, epochs=12, batch=64, patch=(4, 4), lr=3e-3)
    assert losses[-1] < 0.8 * losses[0]


def test_copy_encoder_transfers_trunk_but_not_projection():
    a = models.Backbone(widths=(8, 16), feat_dim=8)
    b = models.Backbone(widths=(8, 16), feat_dim=8)
    ssl.copy_encoder(a, b)
    for (ka, va), (kb, vb) in zip(a.blocks.state_dict().items(), b.blocks.state_dict().items()):
        assert torch.equal(va, vb)
    assert not torch.equal(a.proj.weight, b.proj.weight)


def test_pretrain_with_fewer_windows_than_a_batch_still_trains():
    """Regression: n < batch ran zero steps and reported a loss of 0.0 without warning."""
    X = torch.randn(40, 1, 16, 24)
    bb = models.Backbone(widths=(8, 16), feat_dim=8)
    losses = ssl.pretrain(bb, X, epochs=2, batch=128, patch=(4, 4))
    assert losses[0] > 0
    with pytest.raises(ValueError):
        ssl.pretrain(bb, torch.zeros(0, 1, 16, 24), epochs=1)
