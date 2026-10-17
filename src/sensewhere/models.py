"""Backbone and the three interchangeable classification heads (flat softmax, hierarchical tree, mixture of experts).

Every head maps backbone features to log-probabilities over the same leaf classes, so metrics, adaptation and
class extension work identically for all of them.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Backbone(nn.Module):
    """Compact 2D CNN with BatchNorm over a (1, frequency/Doppler, time) input; global average pooling."""

    def __init__(self, in_ch: int = 1, widths: tuple[int, ...] = (16, 32, 64, 64), feat_dim: int = 64):
        super().__init__()
        blocks, c = [], in_ch
        for w in widths:
            blocks += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2, ceil_mode=True)]
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Linear(c, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x).mean(dim=(2, 3))
        return F.relu(self.proj(h))


class FlatHead(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    @property
    def n_classes(self) -> int:
        return self.fc.out_features

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.fc(f), dim=-1)

    def aux_loss(self) -> torch.Tensor:
        return torch.zeros(())

    @torch.no_grad()
    def extend(self, n_new: int, **_) -> list[int]:
        old = self.fc
        new = nn.Linear(old.in_features, old.out_features + n_new)
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
        nn.init.normal_(new.weight[old.out_features :], std=0.01)
        nn.init.zeros_(new.bias[old.out_features :])
        self.fc = new
        return list(range(old.out_features, old.out_features + n_new))


class HierHead(nn.Module):
    """Tree of local softmax classifiers. `tree` maps an internal node name to its ordered children; names that are
    not keys are leaves and must be listed in `leaves` (their order defines the class index)."""

    def __init__(self, dim: int, tree: dict[str, list[str]], leaves: list[str], root: str = "root"):
        super().__init__()
        self.tree = {k: list(v) for k, v in tree.items()}
        self.leaves = list(leaves)
        self.root = root
        self.dim = dim
        self.nodes = nn.ModuleDict({k: nn.Linear(dim, len(v)) for k, v in self.tree.items()})
        self._check()

    def _check(self) -> None:
        reach = set()

        def walk(n: str) -> None:
            if n in self.tree:
                for c in self.tree[n]:
                    walk(c)
            else:
                reach.add(n)

        walk(self.root)
        if reach != set(self.leaves):
            raise ValueError(f"tree leaves {sorted(reach)} do not match declared leaves {sorted(self.leaves)}")

    @property
    def n_classes(self) -> int:
        return len(self.leaves)

    def _paths(self) -> dict[str, list[tuple[str, int]]]:
        out: dict[str, list[tuple[str, int]]] = {}

        def walk(n: str, path: list[tuple[str, int]]) -> None:
            if n in self.tree:
                for i, c in enumerate(self.tree[n]):
                    walk(c, path + [(n, i)])
            else:
                out[n] = path

        walk(self.root, [])
        return out

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        local = {k: F.log_softmax(m(f), dim=-1) for k, m in self.nodes.items()}
        paths = self._paths()
        cols = []
        for leaf in self.leaves:
            lp = torch.zeros(f.shape[0], device=f.device)
            for node, i in paths[leaf]:
                lp = lp + local[node][:, i]
            cols.append(lp)
        return torch.stack(cols, dim=1)

    def aux_loss(self) -> torch.Tensor:
        return torch.zeros(())

    @torch.no_grad()
    def extend(self, n_new: int, parent: str = "root", names: list[str] | None = None, **_) -> list[int]:
        """Add n_new leaves as children of `parent`; only that node's classifier grows, every other node is untouched."""
        names = names or [f"new{len(self.leaves) + i}" for i in range(n_new)]
        old = self.nodes[parent]
        new = nn.Linear(old.in_features, old.out_features + n_new)
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
        nn.init.normal_(new.weight[old.out_features :], std=0.01)
        nn.init.zeros_(new.bias[old.out_features :])
        self.nodes[parent] = new
        self.tree[parent] = self.tree[parent] + names
        first = len(self.leaves)
        self.leaves = self.leaves + names
        self._check()
        return list(range(first, first + n_new))


class MoEHead(nn.Module):
    """Mixture of experts: a softmax gate mixes the class probabilities of E small expert classifiers.

    top_k < E keeps only the k largest gate weights per sample (renormalised). aux_loss is the Switch-style
    load-balancing term E * sum_e(importance_e * load_e), which is 1 for a perfectly balanced gate."""

    def __init__(self, dim: int, n_classes: int, n_experts: int = 4, top_k: int = 2, hidden: int = 64):
        super().__init__()
        self.dim, self.hidden, self.top_k = dim, hidden, top_k
        self.gate = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList([self._expert(n_classes) for _ in range(n_experts)])
        self._aux = torch.zeros(())

    def _expert(self, n_classes: int) -> nn.Module:
        return nn.Sequential(nn.Linear(self.dim, self.hidden), nn.ReLU(), nn.Linear(self.hidden, n_classes))

    @property
    def n_classes(self) -> int:
        return self.experts[0][-1].out_features

    def gate_weights(self, f: torch.Tensor) -> torch.Tensor:
        g = F.softmax(self.gate(f), dim=-1)
        if self.top_k < g.shape[1]:
            top, idx = g.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(g).scatter_(1, idx, 1.0)
            g = g * mask
            g = g / g.sum(-1, keepdim=True)
        return g

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        g = self.gate_weights(f)
        probs = torch.stack([F.softmax(e(f), dim=-1) for e in self.experts], dim=1)  # B,E,K
        mix = (g.unsqueeze(-1) * probs).sum(1)
        e = g.shape[1]
        importance = g.mean(0)
        load = (g > 0).float().mean(0)
        self._aux = e * (importance * load).sum() / max(1.0, float(self.top_k))
        return torch.log(mix.clamp_min(1e-8))

    def aux_loss(self) -> torch.Tensor:
        return self._aux

    @torch.no_grad()
    def extend(self, n_new: int, add_expert: bool = True, freeze_old: bool = False, **_) -> list[int]:
        """Grow every expert's output layer; optionally add one expert (and a gate output) that old experts leave alone."""
        old_k = self.n_classes
        for i, e in enumerate(self.experts):
            last = e[-1]
            new = nn.Linear(last.in_features, old_k + n_new)
            new.weight[:old_k] = last.weight
            new.bias[:old_k] = last.bias
            nn.init.normal_(new.weight[old_k:], std=0.01)
            nn.init.zeros_(new.bias[old_k:])
            e[-1] = new
            if freeze_old:
                for p in e.parameters():
                    p.requires_grad_(False)
        if add_expert:
            self.experts.append(self._expert(old_k + n_new))
            g = self.gate
            ng = nn.Linear(g.in_features, g.out_features + 1)
            ng.weight[: g.out_features] = g.weight
            ng.bias[: g.out_features] = g.bias
            nn.init.normal_(ng.weight[g.out_features :], std=0.01)
            nn.init.zeros_(ng.bias[g.out_features :])
            self.gate = ng
        return list(range(old_k, old_k + n_new))


class Classifier(nn.Module):
    """Backbone + head. With input_norm each window is standardised (mean 0, std 1) inside forward, so cached windows
    can stay in raw dB and input-space augmentations (e.g. noise-floor transplant) happen before normalisation."""

    def __init__(self, backbone: Backbone, head: nn.Module, input_norm: bool = False):
        super().__init__()
        self.backbone, self.head, self.input_norm = backbone, head, input_norm

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        if self.input_norm:
            dims = tuple(range(1, x.ndim))
            x = (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-6)
        f = self.backbone(x)
        out = self.head(f)
        return (out, f) if return_feats else out


def make_head(kind: str, dim: int, n_classes: int, tree: dict | None = None, leaves: list[str] | None = None, **kw) -> nn.Module:
    if kind == "flat":
        return FlatHead(dim, n_classes)
    if kind == "hier":
        if tree is None or leaves is None:
            raise ValueError("hier head needs tree and leaves")
        return HierHead(dim, tree, leaves)
    if kind == "moe":
        return MoEHead(dim, n_classes, **kw)
    raise ValueError(f"unknown head {kind}")


def n_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


def entropy_from_logp(logp: torch.Tensor) -> torch.Tensor:
    return -(logp.exp() * logp).sum(-1)

