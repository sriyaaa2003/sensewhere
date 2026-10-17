"""Track-contrastive pretraining: two windows of the same track are positives, windows of other tracks are negatives.

The activity is constant within a track, so the pretext task asks the encoder to find what stays the same across time
for a given person and activity (gait rhythm, arm posture) and to ignore what changes (instantaneous phase). Unlike
masked reconstruction it needs no labels but still points at the property the downstream task uses.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import Backbone
from .train import TrainConfig, augment


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE: row i of z1 and row i of z2 are positives; every other row of either view is a negative."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def sample_pairs(track_ids: np.ndarray, batch_tracks: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pick batch_tracks tracks with at least two windows and two different windows from each."""
    uniq, inv = np.unique(track_ids, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    ok = [k for k, m in enumerate(members) if len(m) >= 2]
    if not ok:
        raise ValueError("no track has two windows: track-contrastive pretraining needs positives")
    pick = rng.choice(ok, size=min(batch_tracks, len(ok)), replace=False)
    a, b = [], []
    for k in pick:
        i, j = rng.choice(members[k], size=2, replace=False)
        a.append(i)
        b.append(j)
    return np.array(a), np.array(b)


def pretrain_contrastive(
    backbone: Backbone,
    X: torch.Tensor,
    track_ids: np.ndarray,
    steps: int = 600,
    batch_tracks: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.2,
    proj_dim: int = 64,
    seed: int = 0,
    device: str = "cpu",
    stretch=None,
) -> list[float]:
    """Pretrain the backbone in place with a throw-away projection head; returns the loss every 50 steps."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    proj = nn.Sequential(nn.Linear(backbone.feat_dim, 128), nn.ReLU(), nn.Linear(128, proj_dim))
    backbone, proj = backbone.to(device), proj.to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(proj.parameters()), lr=lr, weight_decay=1e-4)
    cfg = TrainConfig(augment=True)
    losses = []

    def view(x):
        x = augment(x, cfg, g)
        if stretch is not None:
            x = stretch(x, g)
        dims = tuple(range(1, x.ndim))
        return (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-6)

    run = []
    backbone.train()
    for s in range(steps):
        a, b = sample_pairs(track_ids, batch_tracks, rng)
        xa, xb = view(X[a].to(device)), view(X[b].to(device))
        loss = nt_xent(proj(backbone(xa)), proj(backbone(xb)), temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run.append(float(loss.detach()))
        if (s + 1) % 50 == 0:
            losses.append(float(np.mean(run)))
            run = []
    if run:
        losses.append(float(np.mean(run)))
    backbone.cpu()
    return losses
