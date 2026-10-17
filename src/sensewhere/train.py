"""One training loop for every method. Method-specific logic lives in dg.py, adapt.py and models.py."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import dg, floor
from .metrics import balanced_accuracy
from .models import Classifier


@dataclass
class TrainConfig:
    method: str = "erm"            # erm | coral | dann | groupdro  (augmentation is controlled separately, see below)
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_balanced: bool = True
    method_weight: float = 1.0     # CORAL / DANN strength
    dro_step: float = 0.01
    moe_aux_weight: float = 0.01
    seed: int = 0
    device: str = "cpu"
    patience: int = 6              # epochs without source-validation improvement before stopping
    augment: bool = False          # basic time/Doppler shift, gain and noise
    legacy_aug: bool = False       # v1 behaviour: augment for every method except plain erm (kept so v1 results can be rerun)
    floor_transplant: bool = False # swap the site noise-floor profile for another training site's (needs floor_profiles)
    floor_profiles: object = None  # tensor (n_domains, F) of per-site background profiles in the cached windows' units
    freeze_bn: bool = False        # keep BatchNorm in eval mode (statistics frozen) while fine-tuning a frozen backbone
    aug_shift_t: int = 10
    aug_shift_f: int = 2
    aug_scale: float = 0.2
    aug_noise: float = 0.05
    extra: dict = field(default_factory=dict)


def augment(x: torch.Tensor, cfg: TrainConfig, g: torch.Generator) -> torch.Tensor:
    """Time shift, Doppler-bin shift, amplitude jitter and noise on a (B,1,F,T) batch (all label-preserving)."""
    b = x.shape[0]
    out = torch.empty_like(x)
    st = torch.randint(-cfg.aug_shift_t, cfg.aug_shift_t + 1, (b,), generator=g)
    sf = torch.randint(-cfg.aug_shift_f, cfg.aug_shift_f + 1, (b,), generator=g)
    for i in range(b):
        out[i] = torch.roll(x[i], shifts=(int(sf[i]), int(st[i])), dims=(1, 2))
    scale = 1 + cfg.aug_scale * (2 * torch.rand(b, 1, 1, 1, generator=g) - 1)
    out = out * scale.to(x.device) + cfg.aug_noise * torch.randn(out.shape, generator=g).to(x.device)
    return out


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 1024, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, logps = [], []
    for i in range(0, len(X), batch_size):
        lp = model(X[i : i + batch_size].to(device))
        logps.append(lp.cpu())
        preds.append(lp.argmax(1).cpu())
    return torch.cat(preds).numpy(), torch.cat(logps).numpy()


def fit(
    model: Classifier,
    X: torch.Tensor,
    y: np.ndarray,
    domains: np.ndarray,
    Xv: torch.Tensor,
    yv: np.ndarray,
    n_classes: int,
    cfg: TrainConfig,
    n_domains: int | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[Classifier, list[dict]]:
    """Train on (X, y, domains); keep the epoch with the best balanced accuracy on the SOURCE validation set."""
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    dev = cfg.device
    model = model.to(dev)
    n_domains = n_domains or int(domains.max()) + 1
    disc = None
    params = list(model.parameters())
    if cfg.method == "dann":
        disc = dg.DomainDiscriminator(model.backbone.feat_dim, n_domains).to(dev)
        params += list(disc.parameters())
    dro = dg.GroupDRO(n_domains, cfg.dro_step) if cfg.method == "groupdro" else None
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    cw = class_weights(y, n_classes).to(dev) if cfg.class_balanced else None
    yt = torch.from_numpy(y).long()
    dt = torch.from_numpy(domains).long()
    sw = torch.from_numpy(sample_weight).float() if sample_weight is not None else None
    n = len(y)
    bs = min(cfg.batch_size, n)  # fewer windows than one batch would otherwise skip every step
    best, best_state, bad, log = -1.0, None, 0, []
    total_steps = cfg.epochs * max(1, n // bs)
    step = 0
    for ep in range(cfg.epochs):
        model.train()
        if cfg.freeze_bn:
            for mod in model.backbone.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        perm = torch.randperm(n, generator=g)
        run = 0.0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            xb = X[idx].to(dev)
            yb, db = yt[idx].to(dev), dt[idx].to(dev)
            if cfg.floor_transplant:
                prof = cfg.floor_profiles.to(dev)
                donor = floor.donor_indices(db.cpu(), prof.shape[0], g).to(dev)
                xb = floor.transplant(xb, prof[db], prof[donor])
            if cfg.augment or (cfg.legacy_aug and cfg.method != "erm"):
                xb = augment(xb, cfg, g)
            logp, feats = model(xb, return_feats=True)
            per = F.nll_loss(logp, yb, weight=cw, reduction="none")
            if sw is not None:
                per = per * sw[idx].to(dev)
            if dro is not None:
                loss = dro.weighted(per, db)
            else:
                loss = per.mean()
            if cfg.method == "coral":
                loss = loss + cfg.method_weight * dg.coral_loss(feats, db)
            if disc is not None:
                lam = 2 / (1 + np.exp(-10 * step / total_steps)) - 1
                loss = loss + cfg.method_weight * F.cross_entropy(disc(feats, float(lam)), db)
            loss = loss + cfg.moe_aux_weight * model.head.aux_loss().to(dev)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            run += float(loss.detach())
        pv, _ = predict(model, Xv, device=dev)
        ba = balanced_accuracy(yv, pv, n_classes)
        log.append({"epoch": ep, "loss": run / max(1, n // bs), "val_bal_acc": ba})
        if ba > best:
            best, bad, best_state = ba, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is None:  # every validation score was NaN: keep the last weights rather than crash
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), log
