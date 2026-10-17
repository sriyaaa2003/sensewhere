import numpy as np
import pytest

from sensewhere import features


def _blob(n_d=64, n_t=90, centre_fn=lambda t: 0.0, width=0.05):
    v = np.linspace(-1, 1, n_d)
    spec = np.zeros((n_d, n_t))
    for t in range(n_t):
        spec[:, t] = np.exp(-((v - centre_fn(t)) ** 2) / (2 * width**2))
    return spec


def test_centroid_tracks_blob_position():
    f = features.window_features(_blob(centre_fn=lambda t: 0.5))
    assert f["centroid_mean"] == pytest.approx(0.5, abs=0.05)
    assert f["positive_fraction"] > 0.9


def test_cadence_recovers_oscillation_frequency():
    fps, hz = 90.0, 2.0
    spec = _blob(centre_fn=lambda t: 0.3 * np.sin(2 * np.pi * hz * t / fps))
    f = features.window_features(spec, fps=fps)
    assert f["cadence_hz"] == pytest.approx(hz, abs=0.6)
    assert f["cadence_strength"] > 0.3


def test_static_target_is_zero_doppler_heavy():
    f = features.window_features(_blob(centre_fn=lambda t: 0.0, width=0.02))
    assert f["static_fraction"] > 0.5
    assert f["centroid_range"] < 0.05


def test_wider_blob_has_larger_spread():
    narrow = features.window_features(_blob(width=0.03))["spread_mean"]
    wide = features.window_features(_blob(width=0.2))["spread_mean"]
    assert wide > narrow


def test_zero_window_is_finite():
    f = features.window_features(np.zeros((32, 90)))
    assert all(np.isfinite(list(f.values())))


def test_negative_input_rejected():
    with pytest.raises(ValueError):
        features.window_features(-np.ones((8, 8)))


def test_feature_matrix_shape_and_names_are_stable():
    specs = np.stack([_blob(), _blob(centre_fn=lambda t: 0.2)])
    X, names = features.feature_matrix(specs)
    assert X.shape == (2, len(names)) and len(set(names)) == len(names)
