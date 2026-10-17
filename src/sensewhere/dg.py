"""Domain-generalisation ingredients: CORAL, DANN (gradient reversal) and Group-DRO weights."""
from __future__ import annotations

import torch
from torch import nn


def coral_loss(feats: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between the feature covariances of every pair of domains present in the batch."""
    ids = torch.unique(domains)
    covs = []
    for d in ids:
        x = feats[domains == d]
        if len(x) < 2:
            continue
        x = x - x.mean(0, keepdim=True)
        covs.append(x.T @ x / (len(x) - 1))
    if len(covs) < 2:
        return feats.new_zeros(())
    loss, n = feats.new_zeros(()), 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + ((covs[i] - covs[j]) ** 2).mean()
            n += 1
    return loss / n


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class DomainDiscriminator(nn.Module):
    def __init__(self, dim: int, n_domains: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_domains))

    def forward(self, feats: torch.Tensor, lam: float) -> torch.Tensor:
        return self.net(grad_reverse(feats, lam))


class GroupDRO:
    """Exponentiated-gradient weights over per-domain losses (worst-domain emphasis)."""

    def __init__(self, n_domains: int, step: float = 0.01):
        self.q = torch.ones(n_domains) / n_domains
        self.step = step

    def weighted(self, per_sample_loss: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        n = len(self.q)
        dev = per_sample_loss.device
        losses = torch.zeros(n, device=dev)
        present = torch.zeros(n, dtype=torch.bool, device=dev)
        for d in range(n):
            m = domains == d
            if m.any():
                losses[d] = per_sample_loss[m].mean()
                present[d] = True
        q = self.q.to(dev) * torch.exp(self.step * losses.detach() * present)
        q = q / q.sum()
        self.q = q.cpu()
        return (q * losses).sum()
