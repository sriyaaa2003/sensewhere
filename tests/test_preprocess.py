import numpy as np
import pytest

from sensewhere import preprocess as pp

CFG = pp.DopplerConfig()


def _track(t=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((256, t)) + 0.1


def test_slice_window_bounds():
    tr = _track(300)
    assert pp.slice_window(tr, 0.0, CFG).shape == (256, 90)
    assert pp.slice_window(tr, 2.0, CFG).shape == (256, 90)
    assert pp.slice_window(tr, 3.5, CFG) is None  # runs past the end
    assert pp.slice_window(tr, -0.1, CFG) is None


def test_suppress_static_removes_centre_spike():
    x = np.ones((256, 10))
    x[126:131] = 100.0
    out = pp.suppress_static(x, 2)
    assert out.max() == pytest.approx(1.0)
    assert out.shape == x.shape


def test_crop_and_downsample_shape():
    x = np.random.default_rng(0).random((256, 90))
    out = pp.crop_and_downsample(x, 96, 64)
    assert out.shape == (64, 90)


def test_crop_downsample_preserves_mean_of_cropped_region():
    x = np.ones((256, 5)) * 3.0
    assert pp.crop_and_downsample(x, 96, 64).mean() == pytest.approx(3.0)


def test_process_window_normalised_and_float32():
    w = pp.slice_window(_track(), 0.5, CFG)
    out, why = pp.process_doppler_window(w, CFG)
    assert why == "ok" and out.dtype == np.float32 and out.shape == (64, 90)
    assert abs(float(out.mean())) < 1e-4 and float(out.std()) == pytest.approx(1.0, abs=1e-3)


def test_process_window_rejects_nan_constant_and_wrong_length():
    w = pp.slice_window(_track(), 0.0, CFG).copy()
    w[3, 3] = np.nan
    assert pp.process_doppler_window(w, CFG) == (None, "non_finite")
    assert pp.process_doppler_window(np.ones((256, 90)), CFG) == (None, "near_constant")
    assert pp.process_doppler_window(np.ones((256, 80)), CFG) == (None, "wrong_length")


def test_window_result_independent_of_other_windows():
    tr = _track(400)
    a, _ = pp.process_doppler_window(pp.slice_window(tr, 0.5, CFG), CFG)
    tr2 = tr.copy()
    tr2[:, 200:] *= 50  # change a different part of the track
    b, _ = pp.process_doppler_window(pp.slice_window(tr2, 0.5, CFG), CFG)
    assert np.array_equal(a, b)


def test_unknown_normalise_raises():
    w = pp.slice_window(_track(), 0.0, CFG)
    with pytest.raises(ValueError):
        pp.process_doppler_window(w, pp.DopplerConfig(normalise="per_site"))


def test_bvp_resamples_time_and_flags_empty():
    rng = np.random.default_rng(0)
    bvp = np.zeros((20, 20, 12))
    bvp[:, :, 2:9] = rng.random((20, 20, 7))
    out, why = pp.process_bvp(bvp, pp.BVPConfig(n_frames=20))
    assert why == "ok" and out.shape == (20, 20, 20)
    assert pp.process_bvp(np.zeros((20, 20, 12)), pp.BVPConfig()) == (None, "empty")
    bad = bvp.copy()
    bad[0, 0, 0] = np.inf
    assert pp.process_bvp(bad, pp.BVPConfig()) == (None, "non_finite")


def test_robust_normalisation_is_median_centred_and_offset_invariant():
    cfg = pp.DopplerConfig(normalise="robust")
    w = pp.slice_window(_track(), 0.5, cfg)
    a, why = pp.process_doppler_window(w, cfg)
    b, _ = pp.process_doppler_window(w + 7.0, cfg)  # a site-level offset in dB must vanish
    assert why == "ok" and abs(float(np.median(a))) < 1e-4
    assert np.allclose(a, b, atol=1e-4)


def test_none_normalisation_keeps_site_offset():
    cfg = pp.DopplerConfig(normalise="none")
    w = pp.slice_window(_track(), 0.5, cfg)
    a, _ = pp.process_doppler_window(w, cfg)
    b, _ = pp.process_doppler_window(w + 7.0, cfg)
    assert float(b.mean()) - float(a.mean()) == pytest.approx(7.0, abs=1e-3)


def test_default_config_does_not_log_compress():
    assert pp.DopplerConfig().log_compress is False


def test_bvp_with_zero_frames_is_flagged_empty_not_crashed():
    assert pp.process_bvp(np.zeros((20, 20, 0)), pp.BVPConfig()) == (None, "empty")
