"""Masked-reconstruction pretraining of the CNN encoder on unlabeled windows.

Random Doppler-time patches of a (standardised) window are hidden and a light decoder must reconstruct them from the
visible context. The encoder is the classifier's own convolutional trunk, so its weights (and BatchNorm statistics) can
be copied straight into a classifier before supervised fine-tuning.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import Backbone


class MaskedAE(nn.Module):
    def __init__(self, backbone: Backbone, out_ch: int = 1):
        super().__init__()
        self.encoder = backbone.blocks  # shared with the classifier
        widths = [m.out_channels for m in backbone.blocks if isinstance(m, nn.Conv2d)]
        layers, c = [], widths[-1]
        for w in reversed(widths[:-1]):
            layers += [nn.ConvTranspose2d(c, w, 2, stride=2), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        layers += [nn.ConvTranspose2d(c, c, 2, stride=2), nn.ReLU(inplace=True), nn.Conv2d(c, out_ch, 3, padding=1)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        r = self.decoder(z)
        return F.interpolate(r, size=x.shape[-2:], mode="bilinear", align_corners=False)


def standardise(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    return (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-6)


def random_patch_mask(shape: tuple[int, ...], patch: tuple[int, int], ratio: float, g: torch.Generator) -> torch.Tensor:
    """Boolean mask (B,1,F,T), True = hidden. Patches are hidden independently with probability `ratio`."""
    b, _, f, t = shape
    ph, pw = patch
    gh, gw = -(-f // ph), -(-t // pw)
    cells = torch.rand(b, 1, gh, gw, generator=g) < ratio
    m = cells.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)
    return m[:, :, :f, :t]


def pretrain(
    backbone: Backbone,
    X: torch.Tensor,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 1e-3,
    ratio: float = 0.5,
    patch: tuple[int, int] = (8, 9),
    seed: int = 0,
    device: str = "cpu",
) -> list[float]:
    """Pretrain `backbone.blocks` in place; returns the mean masked-reconstruction loss per epoch."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    ae = MaskedAE(backbone).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X)
    if n == 0:
        raise ValueError("no windows to pretrain on")
    batch = min(batch, n)  # fewer windows than one batch would otherwise run zero steps and report zero loss
    losses = []
    for _ in range(epochs):
        ae.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n - batch + 1, batch):
            xb = standardise(X[perm[i : i + batch]].to(device))
            mask = random_patch_mask(xb.shape, patch, ratio, g).to(device)
            rec = ae(xb.masked_fill(mask, 0.0))
            loss = ((rec - xb) ** 2)[mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            cnt += 1
        losses.append(tot / max(cnt, 1))
    backbone.cpu()
    return losses


def copy_encoder(src: Backbone, dst: Backbone) -> None:
    """Copy the pretrained trunk (conv weights and BatchNorm statistics); the projection layer of dst stays fresh."""
    dst.blocks.load_state_dict(copy.deepcopy(src.blocks.state_dict()))
