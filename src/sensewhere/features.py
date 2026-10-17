"""Hand-built descriptors of a micro-Doppler window, used by the gradient-boosting baseline.

Input is a non-negative magnitude (or power) spectrogram of shape (n_doppler, n_time) with the velocity axis
symmetric around zero. The descriptors are physical: torso speed (Doppler centroid), limb swing (spread and
envelope), gait cadence (periodicity of the centroid) and where the energy sits.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES: list[str] | None = None


def _stats(prefix: str, x: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
    }


def window_features(spec: np.ndarray, fps: float = 90.0, envelope_quantile: float = 0.95) -> dict[str, float]:
    """Descriptors of one window. `spec` must be >= 0; zero windows return zeros rather than NaN."""
    if spec.ndim != 2:
        raise ValueError("spec must be (n_doppler, n_time)")
    if (spec < 0).any():
        raise ValueError("spec must be non-negative (use magnitude or power)")
    n_d, n_t = spec.shape
    v = np.linspace(-1.0, 1.0, n_d)  # normalised velocity axis; units cancel across sites
    energy = spec.sum(0)
    safe = np.where(energy > 0, energy, 1.0)
    p = spec / safe  # per-frame distribution over velocity
    centroid = (v[:, None] * p).sum(0)
    spread = np.sqrt(np.maximum(((v[:, None] - centroid[None, :]) ** 2 * p).sum(0), 0.0))
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(0) / np.log(n_d)

    cdf = np.cumsum(p, axis=0)
    upper = v[np.minimum((cdf < envelope_quantile).sum(0), n_d - 1)]
    lower = v[np.minimum((cdf < 1 - envelope_quantile).sum(0), n_d - 1)]

    c0 = centroid - centroid.mean()
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fps)
    mag = np.abs(np.fft.rfft(c0 * np.hanning(n_t)))
    band = (freqs >= 0.5) & (freqs <= 6.0)  # plausible gait and swing cadence in Hz
    if band.any() and mag[band].sum() > 0:
        k = int(np.argmax(np.where(band, mag, 0.0)))
        cadence, strength = float(freqs[k]), float(mag[k] / mag[band].sum())
    else:
        cadence, strength = 0.0, 0.0

    mean_spec = spec.mean(1)
    ms = mean_spec / max(mean_spec.sum(), 1e-12)
    m1 = float((v * ms).sum())
    m2 = float(((v - m1) ** 2 * ms).sum())
    skew = float(((v - m1) ** 3 * ms).sum() / max(m2 ** 1.5, 1e-12))
    kurt = float(((v - m1) ** 4 * ms).sum() / max(m2 ** 2, 1e-12))
    centre = np.abs(v) <= 0.05
    total = max(float(spec.sum()), 1e-12)

    out: dict[str, float] = {}
    out.update(_stats("centroid", centroid))
    out.update(_stats("spread", spread))
    out.update(_stats("entropy", ent))
    out.update(_stats("upper_env", upper))
    out.update(_stats("lower_env", lower))
    out.update(_stats("energy", np.log1p(energy)))
    out["cadence_hz"] = cadence
    out["cadence_strength"] = strength
    out["centroid_range"] = float(centroid.max() - centroid.min())
    out["mean_skew"] = skew
    out["mean_kurt"] = kurt
    out["static_fraction"] = float(spec[centre].sum() / total)
    out["positive_fraction"] = float(spec[v > 0.05].sum() / total)
    out["negative_fraction"] = float(spec[v < -0.05].sum() / total)
    return out


def feature_matrix(specs: np.ndarray, fps: float = 90.0) -> tuple[np.ndarray, list[str]]:
    """Stack descriptors for specs of shape (N, n_doppler, n_time)."""
    rows = [window_features(s, fps) for s in specs]
    names = list(rows[0].keys())
    return np.array([[r[n] for n in names] for r in rows], dtype=np.float32), names
