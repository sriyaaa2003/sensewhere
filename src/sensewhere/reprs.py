"""Physics-motivated input representations for micro-Doppler windows.

A walking person is a torso moving at some velocity (a bright ridge in Doppler) plus limb motion around it. Speed and
the approach angle to the radar move and stretch the ridge but say little about the posture of the arms, which is the
cue that separates walking from carrying or hands in pockets. These tools remove the ridge and simulate angle changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def centroid_track(X: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Per-frame Doppler centroid (in bin units, N x T) of the energy above each window's median floor, smoothed in time."""
    X = np.asarray(X, dtype=np.float32)
    n, f, t = X.shape
    floor = np.median(X, axis=(1, 2), keepdims=True)
    w = np.clip(X - floor, 0, None)
    bins = np.arange(f, dtype=np.float32)[None, :, None]
    c = (w * bins).sum(1) / np.maximum(w.sum(1), 1e-6)
    c = np.where(w.sum(1) > 1e-6, c, f / 2.0)
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / smooth
        pad = smooth // 2
        cp = np.pad(c, ((0, 0), (pad, pad)), mode="edge")
        c = np.stack([np.convolve(row, k, mode="valid") for row in cp])
    return c


def torso_align(X: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Roll every frame so that its smoothed Doppler centroid sits in the middle bin (edges filled with the window floor).

    Returns float32 (N, F, T). The torso velocity trajectory is removed; limb ripple around it is kept."""
    X = np.asarray(X, dtype=np.float32)
    n, f, t = X.shape
    c = centroid_track(X, smooth)
    shift = np.rint(f / 2.0 - c).astype(int)  # positive = move content to higher bins
    floor = np.median(X, axis=(1, 2))
    out = np.empty_like(X)
    for i in range(n):
        for j in range(t):
            s = int(np.clip(shift[i, j], -f + 1, f - 1))
            col = np.full(f, floor[i], dtype=np.float32)
            if s >= 0:
                col[s:] = X[i, : f - s, j]
            else:
                col[: f + s] = X[i, -s:, j]
            out[i, :, j] = col
    return out


def stretch_doppler(x: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """Rescale the Doppler axis of each window about the central bin by its own factor (>1 widens), keeping the shape.

    Simulates a change of approach angle: a person seen more across the beam has Doppler compressed by cos(angle)."""
    b, c, f, t = x.shape
    theta = torch.zeros(b, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = 1.0                    # time axis unchanged
    theta[:, 1, 1] = 1.0 / factors.to(x.dtype)  # sampling grid scale along Doppler: factor > 1 stretches content
    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def concat_context(meta, k: int = 3, step_s: float = 1.0, tol: float = 1e-3):
    """Indices that chain k consecutive non-overlapping windows of the same track (t, t+step, ...).

    meta needs columns file_name, subject, t_start. Returns (first_idx, index_matrix of shape (M, k)); windows without a
    full chain are dropped so no frame is invented."""
    key = (meta.file_name.astype(str) + "|" + meta.subject.astype(str)).to_numpy()
    t = meta.t_start.to_numpy(dtype=float)
    lookup = {(key[i], int(round(t[i] / tol))): i for i in range(len(meta))}
    rows = []
    for i in range(len(meta)):
        chain = [i]
        for j in range(1, k):
            nxt = lookup.get((key[i], int(round((t[i] + j * step_s) / tol))))
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) == k:
            rows.append(chain)
    idx = np.array(rows, dtype=int).reshape(-1, k)
    return idx[:, 0] if len(idx) else np.array([], dtype=int), idx


def build_context_windows(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Concatenate the windows listed in each row of idx along time: (M, F, k*T)."""
    return np.concatenate([np.asarray(X[idx[:, j]], dtype=np.float32) for j in range(idx.shape[1])], axis=2)
