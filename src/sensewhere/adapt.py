"""Source-free adaptation: the source model and unlabeled target windows are the only inputs.

adabn      re-estimate BatchNorm statistics on the target (no gradient steps).
tent       minimise prediction entropy, updating only BatchNorm scale and shift (Wang et al. 2021).
shot_im    information maximisation (low per-sample entropy, high entropy of the mean prediction) over the
           feature extractor with the classification head frozen (the label-free part of SHOT, Liang et al. 2020).

Each returns the adapted copy of the model and a trace of mean entropy per step so collapse (entropy near zero with
a single predicted class) is visible rather than silent.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch import nn

from .models import entropy_from_logp


def _bn_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def adabn(model: nn.Module, batches: Iterable[torch.Tensor], device: str = "cpu") -> tuple[nn.Module, list[float]]:
    m = copy.deepcopy(model).to(device)
    bns = _bn_layers(m)
    if not bns:
        raise ValueError("model has no BatchNorm layers")
    for b in bns:
        b.reset_running_stats()
        b.momentum = None  # cumulative average over all target batches
    m.train()
    trace = []
    for x in batches:
        logp = m(x.to(device))
        trace.append(float(entropy_from_logp(logp).mean()))
    m.eval()
    return m, trace


def _collect_bn_affine(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for b in _bn_layers(model):
        if b.affine:
            params += [b.weight, b.bias]
    return params


def tent(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    lr: float = 1e-3,
    epochs: int = 1,
    device: str = "cpu",
) -> tuple[nn.Module, list[float]]:
    m = copy.deepcopy(model).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    params = _collect_bn_affine(m)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    batches = list(batches)
    m.train()  # normalise with target batch statistics
    trace = []
    for _ in range(epochs):
        for x in batches:
            logp = m(x.to(device))
            loss = entropy_from_logp(logp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            trace.append(float(loss.detach()))
    m.eval()
    return m, trace


def shot_im(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    lr: float = 1e-3,
    epochs: int = 3,
    div_weight: float = 1.0,
    device: str = "cpu",
) -> tuple[nn.Module, list[float]]:
    m = copy.deepcopy(model).to(device)
    for p in m.head.parameters():
        p.requires_grad_(False)
    params = [p for p in m.backbone.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    batches = list(batches)
    m.train()
    trace = []
    for _ in range(epochs):
        for x in batches:
            logp = m(x.to(device))
            ent = entropy_from_logp(logp).mean()
            mean_p = logp.exp().mean(0)
            div = (mean_p * torch.log(mean_p.clamp_min(1e-8))).sum()  # negative entropy of the marginal
            loss = ent + div_weight * div
            opt.zero_grad()
            loss.backward()
            opt.step()
            trace.append(float(ent.detach()))
    m.eval()
    return m, trace


# ----------------------------------------------------------------------------------------------------------------------
# v2 additions: a fair SHOT, track voting and track-consistency self-training (TCST)
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .train import predict  # noqa: E402


def track_vote(logp: np.ndarray, track_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average class probabilities over every track and give each window its track's label.

    Valid whenever the tracker supplies track ids and one track holds one activity. Returns (window labels, window
    confidence = the track's mean maximum probability)."""
    p = np.exp(logp)
    uniq, inv = np.unique(track_ids, return_inverse=True)
    sums = np.zeros((len(uniq), p.shape[1]))
    np.add.at(sums, inv, p)
    counts = np.bincount(inv).astype(float)[:, None]
    mean_p = sums / counts
    return mean_p.argmax(1)[inv], mean_p.max(1)[inv]


def shot(
    model: nn.Module,
    X: torch.Tensor,
    batch: int = 128,
    lr: float = 1e-4,
    epochs: int = 5,
    div_weight: float = 1.0,
    pl_weight: float = 0.3,
    params: str = "bn",
    device: str = "cpu",
) -> tuple[nn.Module, list[float]]:
    """SHOT with its pseudo-labelling stage and conservative defaults: information maximisation plus cluster-based
    pseudo-labels, updating only BatchNorm scale/shift (params='bn') or the whole backbone (params='all'); the head
    stays frozen. The v1 function shot_im (lr 1e-3, whole backbone, no pseudo-labels) is kept unchanged for the v1 results."""
    m = copy.deepcopy(model).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    if params == "bn":
        train_params = _collect_bn_affine(m)
    elif params == "all":
        train_params = list(m.backbone.parameters())
    else:
        raise ValueError(params)
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train_params, lr=lr)
    n = len(X)
    trace = []
    for _ in range(epochs):
        # pseudo-labels: probability-weighted class centroids of target features, nearest centroid in cosine distance
        m.eval()
        with torch.no_grad():
            feats, probs = [], []
            for i in range(0, n, 1024):
                lp, f = m(X[i : i + 1024].to(device), return_feats=True)
                feats.append(F.normalize(f, dim=1))
                probs.append(lp.exp())
            feats, probs = torch.cat(feats), torch.cat(probs)
            cent = (probs.T @ feats) / probs.sum(0).clamp_min(1e-6)[:, None]
            pseudo = (feats @ F.normalize(cent, dim=1).T).argmax(1)
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n - 1, batch):
            idx = perm[i : i + batch]
            if len(idx) < 2:
                continue
            lp = m(X[idx].to(device))
            ent = entropy_from_logp(lp).mean()
            mp = lp.exp().mean(0)
            div = (mp * torch.log(mp.clamp_min(1e-8))).sum()
            ce = F.nll_loss(lp, pseudo[idx].to(device))
            loss = ent + div_weight * div + pl_weight * ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            trace.append(float(ent.detach()))
    m.eval()
    return m, trace


def tcst(
    model: nn.Module,
    X: torch.Tensor,
    track_ids: np.ndarray,
    rounds: int = 3,
    keep: float = 0.6,
    batch: int = 128,
    lr: float = 1e-3,
    steps_per_round: int = 1,
    device: str = "cpu",
) -> tuple[nn.Module, list[float]]:
    """Track-consistency self-training (source-free: only unlabeled target windows and their track ids are used).

    Each round: (1) predict all windows and average the probabilities per track, (2) keep the `keep` most confident
    tracks within each predicted class (so no class can starve), (3) train BatchNorm scale/shift with cross-entropy on
    the track labels of the kept windows, pulling every window of a track toward its track's consensus. The first step
    re-estimates BatchNorm statistics on the target (AdaBN)."""
    m, _ = adabn(model, [X[i : i + batch] for i in range(0, len(X), batch)], device)
    for p in m.parameters():
        p.requires_grad_(False)
    params = _collect_bn_affine(m)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    trace = []
    n = len(X)
    for _ in range(rounds):
        _, logp = predict(m, X, device=device)
        labels, conf = track_vote(logp, track_ids)
        keep_mask = np.zeros(n, dtype=bool)
        for c in np.unique(labels):
            sel = np.flatnonzero(labels == c)
            thr = np.quantile(conf[sel], 1 - keep)
            keep_mask[sel[conf[sel] >= thr]] = True
        idx_all = np.flatnonzero(keep_mask)
        if len(idx_all) < 2:
            break
        yt = torch.from_numpy(labels).long()
        m.eval()  # statistics are fixed after the AdaBN step; only scale and shift move
        for _ in range(steps_per_round):
            perm = np.random.default_rng(len(trace)).permutation(idx_all)
            for i in range(0, len(perm), batch):
                idx = torch.from_numpy(perm[i : i + batch])
                lp = m(X[idx].to(device))
                loss = F.nll_loss(lp, yt[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                trace.append(float(loss.detach()))
    m.eval()
    return m, trace
