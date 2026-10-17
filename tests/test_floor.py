import numpy as np
import pytest
import torch

from sensewhere import floor


def _windows(offset, n=40, F=16, T=20, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, F)[None, :, None]
    return (rng.normal(0, 1, (n, F, T)) + base * 0 + offset[None, :, None]).astype(np.float32)


def test_profile_recovers_site_offset_without_labels():
    off = np.linspace(5, 9, 16).astype(np.float32)
    p = floor.site_profile(_windows(off))
    assert np.allclose(p, off, atol=0.15)


def test_calibration_removes_site_difference():
    offA, offB = np.full(16, 5.0, np.float32), np.linspace(8, 12, 16).astype(np.float32)
    XA, XB = _windows(offA, seed=1), _windows(offB, seed=2)
    X = np.concatenate([XA, XB]); sites = np.array(["A"] * 40 + ["B"] * 40)
    prof = floor.site_profiles(X, sites)
    C = floor.calibrate(X, sites, prof)
    assert abs(C[:40].mean() - C[40:].mean()) < 0.1
    assert abs(X[:40].mean() - X[40:].mean()) > 2


def test_profile_is_label_free_and_uses_only_the_given_site():
    off = np.full(16, 3.0, np.float32)
    X = _windows(off)
    X2 = np.concatenate([X, _windows(off + 100, seed=9)])
    sites = np.array(["A"] * 40 + ["B"] * 40)
    assert np.allclose(floor.site_profiles(X2, sites)["A"], floor.site_profile(X))


def test_transplant_moves_profile_exactly():
    x = torch.randn(3, 1, 16, 10) + 4.0
    own, donor = torch.full((3, 16), 4.0), torch.full((3, 16), 9.0)
    y = floor.transplant(x, own, donor)
    assert torch.allclose(y, x + 5.0)


def test_donor_never_self_when_requested_and_in_range():
    g = torch.Generator().manual_seed(0)
    dom = torch.randint(0, 3, (500,), generator=g)
    d = floor.donor_indices(dom, 3, g, include_self=False)
    assert (d != dom).all() and d.min() >= 0 and d.max() < 3
