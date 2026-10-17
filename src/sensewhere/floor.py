"""Noise-floor tools: per-site background profiles, calibration, and noise-floor transplant augmentation.

A radar window in dB is roughly signal + a site-specific background (receiver noise floor, clutter). The background is
additive in dB and the same for every window of a site, so it can be estimated from unlabeled windows alone:

* calibration     subtract a site's own profile (done for every training site and, at deployment, for the new site
                  from its unlabeled windows only: no labels, no gradient steps)
* transplant      during training replace a window's site profile by another site's profile, so the classifier sees
                  every class under every site's background (a physically motivated style swap)
"""
from __future__ import annotations

import numpy as np
import torch


def site_profile(X: np.ndarray, q: float = 50.0) -> np.ndarray:
    """Per-Doppler-bin background of a set of windows (N, F, T): the q-th percentile over windows and time. Labels are not used."""
    return np.percentile(np.asarray(X, dtype=np.float32), q, axis=(0, 2)).astype(np.float32)


def site_profiles(X: np.ndarray, sites: np.ndarray, q: float = 50.0) -> dict[str, np.ndarray]:
    return {s: site_profile(X[sites == s], q) for s in sorted(set(sites.tolist()))}


def calibrate(X: np.ndarray, sites: np.ndarray, profiles: dict[str, np.ndarray]) -> np.ndarray:
    """Subtract each window's own site profile; returns float32 (N, F, T)."""
    out = np.empty(X.shape, dtype=np.float32)
    for s, p in profiles.items():
        m = sites == s
        out[m] = np.asarray(X[m], dtype=np.float32) - p[None, :, None]
    return out


def transplant(x: torch.Tensor, own: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    """x (B,1,F,T) in dB; own/donor (B,F): swap the site background."""
    return x - own[:, None, :, None] + donor[:, None, :, None]


def donor_indices(dom: torch.Tensor, n_domains: int, g: torch.Generator, include_self: bool = True) -> torch.Tensor:
    """Random donor domain per sample (uniform over domains; optionally never the sample's own)."""
    d = torch.randint(0, n_domains, dom.shape, generator=g)
    if not include_self and n_domains > 1:
        same = d == dom
        d = torch.where(same, (d + 1 + torch.randint(0, n_domains - 1, dom.shape, generator=g)) % n_domains, d)
    return d
