import numpy as np
import pandas as pd
import pytest
import torch

from sensewhere import reprs


def _ridge(f=64, t=40, centre=20.0, width=1.5, floor=10.0, noise=0.1, seed=0, ripple=0.0, n=1):
    rng = np.random.default_rng(seed)
    bins = np.arange(f)[None, :, None]
    c = centre + ripple * np.sin(np.arange(t) / 3.0)[None, None, :]
    x = floor + 30 * np.exp(-((bins - c) ** 2) / (2 * width**2)) + noise * rng.normal(size=(n, f, t))
    return x.astype(np.float32)


def test_centroid_tracks_the_ridge_position():
    c = reprs.centroid_track(_ridge(centre=20.0))
    assert abs(c.mean() - 20.0) < 1.0


def test_torso_align_puts_the_ridge_in_the_middle_for_any_speed():
    a = reprs.torso_align(_ridge(centre=12.0))
    b = reprs.torso_align(_ridge(centre=45.0, seed=1))
    ca, cb = reprs.centroid_track(a).mean(), reprs.centroid_track(b).mean()
    assert abs(ca - 32) < 1.5 and abs(cb - 32) < 1.5


def test_torso_align_keeps_limb_ripple_but_removes_torso_trajectory():
    # a ridge whose centre drifts a lot: after alignment the centroid trajectory is flat
    f, t = 64, 40
    rng = np.random.default_rng(0)
    bins = np.arange(f)[None, :, None]
    drift = np.linspace(15, 40, t)[None, None, :]
    x = (10 + 30 * np.exp(-((bins - drift) ** 2) / 8) + 0.1 * rng.normal(size=(1, f, t))).astype(np.float32)
    raw_range = np.ptp(reprs.centroid_track(x))
    aligned_range = np.ptp(reprs.centroid_track(reprs.torso_align(x)))
    assert raw_range > 15 and aligned_range < 3


def test_torso_align_shapes_dtype_and_edge_fill_is_the_floor():
    x = _ridge(centre=5.0, n=3)
    a = reprs.torso_align(x)
    assert a.shape == x.shape and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert abs(float(np.median(a)) - float(np.median(x))) < 2.0


def test_torso_align_blank_window_does_not_crash():
    a = reprs.torso_align(np.full((1, 32, 10), 5.0, dtype=np.float32))
    assert np.allclose(a, 5.0)


def test_stretch_identity_and_direction():
    x = torch.from_numpy(_ridge(centre=32.0, width=2.0)).unsqueeze(1)
    same = reprs.stretch_doppler(x, torch.tensor([1.0]))
    assert torch.allclose(same, x, atol=1e-3)
    wide = reprs.stretch_doppler(x, torch.tensor([2.0]))
    narrow = reprs.stretch_doppler(x, torch.tensor([0.5]))
    spread = lambda z: (z[0, 0].mean(1) > 20).float().sum()
    assert spread(wide) > spread(x) > spread(narrow)


def test_context_chain_requires_consecutive_windows_and_drops_gaps():
    rows = []
    for tstart in [0.0, 0.25, 1.0, 1.25, 2.0, 2.25, 4.0]:  # the last window has no successors; 0.5 is missing
        rows.append(dict(file_name="f", subject=1, t_start=tstart))
    rows += [dict(file_name="f", subject=2, t_start=0.0), dict(file_name="f", subject=2, t_start=1.0), dict(file_name="f", subject=2, t_start=2.0)]
    meta = pd.DataFrame(rows)
    first, idx = reprs.concat_context(meta, k=3)
    got = {tuple(meta.t_start.iloc[r].tolist()) for r in idx}
    assert (0.0, 1.0, 2.0) in got and (0.25, 1.25, 2.25) in got
    assert all(len(set(meta.subject.iloc[r])) == 1 for r in idx)  # chains never cross subjects
    assert len(idx) == 3  # two chains for subject 1 plus one for subject 2


def test_context_windows_have_k_times_the_length():
    X = np.random.default_rng(0).random((6, 8, 10)).astype(np.float32)
    idx = np.array([[0, 1, 2], [3, 4, 5]])
    out = reprs.build_context_windows(X, idx)
    assert out.shape == (2, 8, 30)
    assert np.array_equal(out[0, :, 10:20], X[1])


def test_padded_context_keeps_every_window_and_repeats_the_last_available_one():
    rows = [dict(file_name="f", subject=1, t_start=t) for t in (0.0, 1.0, 5.0)]
    meta = pd.DataFrame(rows)
    idx = reprs.context_indices_padded(meta, k=3)
    assert idx.shape == (3, 3)
    assert idx[0].tolist() == [0, 1, 1]      # 2.0 missing: repeat window at 1.0
    assert idx[1].tolist() == [1, 1, 1]
    assert idx[2].tolist() == [2, 2, 2]
    assert reprs.padded_share(idx) == pytest.approx(5 / 6)


def test_padded_context_never_crosses_tracks():
    rows = [dict(file_name="f", subject=1, t_start=0.0), dict(file_name="f", subject=2, t_start=1.0)]
    idx = reprs.context_indices_padded(pd.DataFrame(rows), k=2)
    assert idx[0].tolist() == [0, 0]
