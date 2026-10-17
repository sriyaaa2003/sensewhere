"""Prototype (nearest-class-mean) classification and extension.

Adding a class needs no classifier retraining, no replay of old windows and no decision boundary to learn: the old
classes are represented by one stored mean embedding each (a few hundred numbers of memory), and a new class is added
by averaging the embeddings of its few labeled windows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def embed(model: nn.Module, X: torch.Tensor, batch: int = 1024, device: str = "cpu") -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        _, f = model(X[i : i + batch].to(device), return_feats=True)
        out.append(F.normalize(f, dim=1).cpu())
    return torch.cat(out)


def class_means(emb: torch.Tensor, y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Mean unit-norm embedding per class (rows of absent classes are zero and can never win)."""
    means = torch.zeros(n_classes, emb.shape[1])
    yt = torch.from_numpy(np.asarray(y)).long()
    for c in range(n_classes):
        m = yt == c
        if m.any():
            means[c] = emb[m].mean(0)
    return F.normalize(means, dim=1)


def ncm_predict(emb: torch.Tensor, means: torch.Tensor) -> np.ndarray:
    return (emb @ means.T).argmax(1).numpy()


def extend_means(means: torch.Tensor, emb_new: torch.Tensor) -> torch.Tensor:
    """Append one new class prototype computed from a handful of labeled windows (existing prototypes are untouched)."""
    new = F.normalize(emb_new.mean(0, keepdim=True), dim=1)
    return torch.cat([means, new], dim=0)
