"""Track-level models: a track (one tracked person) is a bag of 1-second windows that share one activity label.

Window embeddings from a shared CNN are pooled by gated attention (multiple-instance learning) or by a GRU over time,
and the track label is predicted directly, so training matches the deployment unit and uninformative windows (a pause,
an occluded stretch) can be down-weighted by the model instead of being averaged in.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import balanced_accuracy
from .models import Backbone


class AttnPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018): weights a_i = softmax(w^T (tanh(V h_i) * sigmoid(U h_i)))."""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        a = F.softmax(a, dim=1)
        return (a.unsqueeze(-1) * h).sum(1), a


class GRUPool(nn.Module):
    """Bidirectional GRU over the time-ordered window embeddings; masked mean of its outputs."""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # padded windows are never fed to the GRU (packed sequences), so the backward direction cannot see them either
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        o, _ = self.gru(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=h.shape[1])
        m = mask.unsqueeze(-1).float()
        return (o * m).sum(1) / m.sum(1).clamp_min(1.0), mask.float() / mask.float().sum(1, keepdim=True).clamp_min(1.0)


class TrackModel(nn.Module):
    def __init__(self, backbone: Backbone, n_classes: int, pool: str = "attn", input_norm: bool = True):
        super().__init__()
        self.backbone, self.input_norm = backbone, input_norm
        d = backbone.feat_dim
        if pool == "attn":
            self.pool, od = AttnPool(d), d
        elif pool == "gru":
            self.pool = GRUPool(d)
            od = self.pool.out_dim
        else:
            raise ValueError(pool)
        self.head = nn.Linear(od, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm:
            dims = tuple(range(1, x.ndim))
            x = (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-6)
        return self.backbone(x)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """bags (B, K, 1, F, T), mask (B, K) -> log-probabilities (B, C) and window weights (B, K)."""
        b, k = bags.shape[:2]
        h = self.embed(bags.flatten(0, 1)).view(b, k, -1)
        z, w = self.pool(h, mask)
        return F.log_softmax(self.head(z), dim=-1), w


@dataclass
class TrackIndex:
    """Windows grouped by track and ordered in time."""

    tracks: list[str]
    members: list[np.ndarray]   # window indices per track, time-ordered
    labels: np.ndarray          # one label per track
    track_of: np.ndarray        # track id per window (index into tracks)


def index_tracks(track_keys: np.ndarray, t_start: np.ndarray, y: np.ndarray) -> TrackIndex:
    uniq, inv = np.unique(track_keys, return_inverse=True)
    members, labels = [], []
    for k in range(len(uniq)):
        idx = np.flatnonzero(inv == k)
        idx = idx[np.argsort(t_start[idx], kind="stable")]
        members.append(idx)
        ys = np.unique(y[idx])
        if len(ys) != 1:
            raise ValueError(f"track {uniq[k]} holds {len(ys)} labels; the track model assumes one label per track")
        labels.append(int(ys[0]))
    return TrackIndex(list(uniq), members, np.array(labels), inv)


def make_bags(X: torch.Tensor, ti: TrackIndex, track_ids: np.ndarray, k: int, rng: np.random.Generator | None, device: str = "cpu"):
    """Stack k windows per track (a random time-ordered subset when rng is given, otherwise the first k) with a validity mask."""
    bags = torch.zeros(len(track_ids), k, *X.shape[1:])
    mask = torch.zeros(len(track_ids), k, dtype=torch.bool)
    for r, t in enumerate(track_ids):
        idx = ti.members[t]
        if len(idx) > k:
            idx = np.sort(rng.choice(idx, size=k, replace=False)) if rng is not None else idx[:k]
        bags[r, : len(idx)] = X[idx]
        mask[r, : len(idx)] = True
    return bags.to(device), mask.to(device)


@torch.no_grad()
def predict_tracks(model: TrackModel, X: torch.Tensor, ti: TrackIndex, k_max: int = 40, device: str = "cpu", chunk: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(log-probabilities per track, window-level labels broadcast from the track prediction)."""
    model.eval().to(device)
    out = []
    for i in range(0, len(ti.tracks), chunk):
        ids = np.arange(i, min(i + chunk, len(ti.tracks)))
        bags, mask = make_bags(X, ti, ids, k_max, None, device)
        lp, _ = model(bags, mask)
        out.append(lp.cpu())
    lp = torch.cat(out).numpy()
    return lp, lp.argmax(1)[ti.track_of]


def fit_tracks(
    model: TrackModel,
    X: torch.Tensor,
    ti: TrackIndex,
    Xv: torch.Tensor,
    tiv: TrackIndex,
    n_classes: int,
    epochs: int = 25,
    tracks_per_batch: int = 16,
    k: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    seed: int = 0,
    device: str = "cpu",
    augment_fn=None,
) -> tuple[TrackModel, list[dict]]:
    """Train on track bags with class-balanced cross-entropy; keep the epoch with the best balanced accuracy on the
    SOURCE validation tracks. augment_fn (optional) maps a batch of windows (N,1,F,T) to augmented windows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    counts = np.bincount(ti.labels, minlength=n_classes).astype(float)
    cw = torch.tensor(np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.from_numpy(ti.labels).long()
    n_tr = len(ti.tracks)
    best, best_state, bad, log = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_tr)
        run, steps = 0.0, 0
        for i in range(0, max(n_tr - tracks_per_batch + 1, 1), tracks_per_batch):
            ids = perm[i : i + tracks_per_batch]
            bags, mask = make_bags(X, ti, ids, k, rng, device)
            if augment_fn is not None:
                b, kk = bags.shape[:2]
                bags = augment_fn(bags.flatten(0, 1)).view(b, kk, *bags.shape[2:])
            lp, _ = model(bags, mask)
            loss = F.nll_loss(lp, yt[ids].to(device), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach())
            steps += 1
        lp_v, _ = predict_tracks(model, Xv, tiv, device=device)
        ba = balanced_accuracy(tiv.labels, lp_v.argmax(1), n_classes)
        log.append({"epoch": ep, "loss": run / max(steps, 1), "val_bal_acc": ba})
        if ba > best:
            best, bad, best_state = ba, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), log
