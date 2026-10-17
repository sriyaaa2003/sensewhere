"""Window preprocessing for micro-Doppler spectrograms (DopplerWild) and velocity-profile sequences (Widar BVP).

Every step is a pure function of one window plus a validated config, so a window processed at one site never depends
on statistics of another site. Quality control returns a reason string instead of silently dropping data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DopplerConfig:
    fps: float = 90.0
    window_s: float = 1.0
    log_compress: bool = False       # DopplerWild values are already dB-like (1.4 to 78); log1p again would only flatten them
    suppress_bins: int = 2           # zero-Doppler half-width (bins) whose energy is replaced by its neighbours
    crop_half_bins: int = 96         # keep +-crop_half_bins around zero Doppler (from 128 each side)
    out_doppler: int = 64            # height after area downsampling
    normalise: str = "window"        # window (mean/std) | robust (median/MAD) | none
    min_std: float = 1e-6            # windows flatter than this are dropped as near-constant
    max_occlusion_s: float = 0.5     # drop windows occluded longer than this

    @property
    def n_frames(self) -> int:
        return int(round(self.fps * self.window_s))


def slice_window(track: np.ndarray, t_start: float, cfg: DopplerConfig) -> np.ndarray | None:
    """Cut the window named by the CSV. Returns None when the track is too short (caller records the reason)."""
    a = int(round(t_start * cfg.fps))
    b = a + cfg.n_frames
    if a < 0 or b > track.shape[1]:
        return None
    return track[:, a:b]


def suppress_static(spec: np.ndarray, half_width: int) -> np.ndarray:
    """Replace the centre band by the mean of the two bins just outside it (static clutter removal)."""
    if half_width <= 0:
        return spec
    n = spec.shape[0]
    c = n // 2
    lo, hi = c - half_width, c + half_width + 1
    out = spec.copy()
    fill = 0.5 * (spec[lo - 1] + spec[hi]) if lo > 0 and hi < n else np.zeros(spec.shape[1])
    out[lo:hi] = fill[None, :]
    return out


def crop_and_downsample(spec: np.ndarray, crop_half_bins: int, out_doppler: int) -> np.ndarray:
    n = spec.shape[0]
    c = n // 2
    lo, hi = max(0, c - crop_half_bins), min(n, c + crop_half_bins)
    sub = spec[lo:hi]
    h = sub.shape[0]
    if h % out_doppler != 0:
        trim = h - (h // out_doppler) * out_doppler
        sub = sub[trim // 2 : h - (trim - trim // 2)]
        h = sub.shape[0]
    factor = h // out_doppler
    return sub.reshape(out_doppler, factor, -1).mean(1)


def process_doppler_window(window: np.ndarray, cfg: DopplerConfig) -> tuple[np.ndarray | None, str]:
    """(tensor of shape (out_doppler, n_frames) float32, 'ok') or (None, reason)."""
    if window.shape[1] != cfg.n_frames:
        return None, "wrong_length"
    if not np.isfinite(window).all():
        return None, "non_finite"
    x = np.abs(window).astype(np.float64)
    x = suppress_static(x, cfg.suppress_bins)
    if cfg.log_compress:
        x = np.log1p(x)
    x = crop_and_downsample(x, cfg.crop_half_bins, cfg.out_doppler)
    if x.std() < cfg.min_std:
        return None, "near_constant"
    if cfg.normalise == "window":
        x = (x - x.mean()) / x.std()
    elif cfg.normalise == "robust":
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        x = (x - med) / max(mad, 1e-6)
    elif cfg.normalise != "none":
        raise ValueError(f"unknown normalise {cfg.normalise}")
    return x.astype(np.float32), "ok"


@dataclass(frozen=True)
class BVPConfig:
    n_frames: int = 20
    log_compress: bool = False
    normalise: str = "window"
    min_std: float = 1e-9


def process_bvp(bvp: np.ndarray, cfg: BVPConfig) -> tuple[np.ndarray | None, str]:
    """Velocity profile (20, 20, T) -> (T', 20, 20) with T' = cfg.n_frames by resampling in time; flattened later by the model.

    The 2D velocity grid is kept as an image per frame; the time axis becomes the channel axis.
    """
    if bvp.ndim != 3:
        return None, "bad_ndim"
    if not np.isfinite(bvp).all():
        return None, "non_finite"
    t = bvp.shape[2]
    if t == 0:
        return None, "empty"
    nonzero = (bvp.reshape(-1, t).sum(0) > 0)
    if nonzero.sum() < 3:
        return None, "empty"
    x = bvp[:, :, nonzero]
    idx = np.linspace(0, x.shape[2] - 1, cfg.n_frames).round().astype(int)
    x = x[:, :, idx].transpose(2, 0, 1).astype(np.float64)
    if cfg.log_compress:
        x = np.log1p(x)
    if x.std() < cfg.min_std:
        return None, "near_constant"
    if cfg.normalise == "window":
        x = (x - x.mean()) / x.std()
    return x.astype(np.float32), "ok"
