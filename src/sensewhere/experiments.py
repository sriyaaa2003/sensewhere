"""Experiment cells: one function per question, each returning plain result rows plus the predictions needed for bootstrap.

Everything takes arrays and a meta frame, so the same code runs on the real cache on Kaggle and on a tiny synthetic cache in tests.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import adapt, data, models, train
from .metrics import balanced_accuracy, macro_f1
from .splits import Split

LEVELS = {
    "load": data.LOAD_CLASSES,
    "atomic": data.base_hierarchy().leaves,
}


def encode(meta: pd.DataFrame, level: str, classes: list[str] | None = None) -> np.ndarray:
    col = "label_load" if level == "load" else "label_atomic"
    idx = {c: i for i, c in enumerate(classes or LEVELS[level])}
    unknown = set(meta[col]) - set(idx)
    if unknown:
        raise ValueError(f"labels outside the {level} class list: {sorted(unknown)}")
    return meta[col].map(idx).to_numpy()


def build_model(head: str, level: str, feat_dim: int = 64, n_experts: int = 4, top_k: int = 2, in_ch: int = 1, n_classes: int | None = None, hierarchy: data.Hierarchy | None = None, input_norm: bool = False, backbone: str = "cnn") -> models.Classifier:
    if backbone == "cnn":
        bb = models.Backbone(in_ch=in_ch, widths=(16, 32, 64, 64), feat_dim=feat_dim)
    else:
        bb = models.TorchvisionBackbone(backbone, in_ch=in_ch, feat_dim=feat_dim)
    h = hierarchy or data.base_hierarchy()
    n = n_classes or len(LEVELS[level])
    if head == "hier" and level != "atomic" and hierarchy is None:
        raise ValueError("the hierarchical head needs the atomic level or an explicit hierarchy")
    return models.Classifier(bb, models.make_head(head, feat_dim, n, tree=h.tree, leaves=h.leaves, n_experts=n_experts, top_k=top_k), input_norm=input_norm)


def score(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict:
    return {"bal_acc": balanced_accuracy(y, p, n_classes), "macro_f1": macro_f1(y, p, n_classes), "n": int(len(y))}


@dataclass
class Cell:
    """Tensors for one leave-one-location-out split."""

    split: Split
    Xtr: torch.Tensor
    ytr: np.ndarray
    dtr: np.ndarray
    Xva: torch.Tensor
    yva: np.ndarray
    Xte: torch.Tensor
    yte: np.ndarray
    subj_te: np.ndarray
    n_domains: int
    domain_sites: list[str] | None = None   # source sites in domain-index order
    tracks_te: np.ndarray | None = None     # track key (recording + subject) of every target window


def make_cell(X, meta: pd.DataFrame, split: Split, level: str, device: str = "cpu", classes: list[str] | None = None) -> Cell:
    y = encode(meta, level, classes)
    sites = sorted(meta.location.unique())
    src_sites = [s for s in sites if s != split.target]
    dom = meta.location.map({s: i for i, s in enumerate(src_sites)}).fillna(-1).astype(int).to_numpy()
    tt = lambda idx: data.to_tensor(X, idx)
    return Cell(
        split,
        tt(split.train_idx),
        y[split.train_idx],
        dom[split.train_idx],
        tt(split.val_idx),
        y[split.val_idx],
        tt(split.target_idx),
        y[split.target_idx],
        meta.subject.to_numpy()[split.target_idx],
        len(src_sites),
        src_sites,
        (meta.file_name.astype(str) + "|" + meta.subject.astype(str)).to_numpy()[split.target_idx] if "file_name" in meta else None,
    )


def run_method(cell: Cell, level: str, head: str, method: str, seed: int, epochs: int, device: str = "cpu", in_ch: int = 1, n_classes: int | None = None, hierarchy: data.Hierarchy | None = None, input_norm: bool = False, profiles: dict | None = None, init: models.Backbone | None = None,
               backbone: str = "cnn", head_kw: dict | None = None, aux_y: np.ndarray | None = None, aux_classes: int = 0, **kw):
    """Train one model; return (model, predictions on target, log).

    profiles: {site: per-bin background vector} for noise-floor transplant (rows follow cell.domain_sites).
    init: a pretrained Backbone whose convolutional trunk initialises the classifier."""
    n = n_classes or len(LEVELS[level])
    torch.manual_seed(seed)
    m = build_model(head, level, in_ch=in_ch, n_classes=n_classes, hierarchy=hierarchy, input_norm=input_norm, backbone=backbone, **(head_kw or {}))
    if init is not None:
        from .mae import copy_encoder
        copy_encoder(init, m.backbone)
    if profiles is not None:
        kw["floor_profiles"] = torch.from_numpy(np.stack([profiles[s_] for s_ in cell.domain_sites])).float()
    cfg = train.TrainConfig(method=method, epochs=epochs, seed=seed, device=device, **kw)
    m, log = train.fit(m, cell.Xtr, cell.ytr, cell.dtr, cell.Xva, cell.yva, n, cfg, n_domains=cell.n_domains, aux_y=aux_y, aux_classes=aux_classes)
    pred, _ = train.predict(m, cell.Xte, device=device)
    return m, pred, log


def adapt_rows(model, cell: Cell, level: str, device: str = "cpu", batch: int = 128, tent_epochs=(1, 3, 10), shot_epochs=(3, 10), lr: float = 1e-3):
    """Apply every source-free method to one source model using only the unlabeled target windows.

    Returns {name: (pred, entropy_trace)}; `name` encodes the method and its step budget."""
    out = {}
    batches = [cell.Xte[i : i + batch] for i in range(0, len(cell.Xte), batch)]
    m, tr = adapt.adabn(model, batches, device)
    out["adabn"] = (train.predict(m, cell.Xte, device=device)[0], tr)
    for e in tent_epochs:
        m, tr = adapt.tent(model, batches, lr=lr, epochs=e, device=device)
        out[f"tent_e{e}"] = (train.predict(m, cell.Xte, device=device)[0], tr)
    for e in shot_epochs:
        m, tr = adapt.shot_im(model, batches, lr=lr, epochs=e, device=device)
        out[f"shot_e{e}"] = (train.predict(m, cell.Xte, device=device)[0], tr)
    return out


def collapse_share(pred: np.ndarray, n_classes: int) -> float:
    """Share of target windows given to the single most-predicted class (1.0 means total collapse)."""
    return float(np.bincount(pred, minlength=n_classes).max() / max(len(pred), 1))


def extend_and_finetune(
    base: models.Classifier,
    head_kind: str,
    cell: Cell,
    new_X: torch.Tensor,
    new_y_local: np.ndarray,
    new_val_X: torch.Tensor,
    new_val_y_local: np.ndarray,
    n_new: int,
    replay_frac: float,
    seed: int,
    epochs: int = 8,
    device: str = "cpu",
    parent: str = "dual",
    names: list[str] | None = None,
    freeze_backbone: bool = True,
    freeze_bn: bool | None = None,
):
    """Add n_new classes to a trained model and fine-tune on the new-class windows plus a replay slice of old data.

    new_y_local is in 0..n_new-1; labels become n_old + local. The backbone is frozen by default so that only
    head parameters carry the extension (the quantity under study); replay_frac in [0,1] is the share of OLD
    training windows kept (0 means none)."""
    m = copy.deepcopy(base)
    n_old = m.head.n_classes
    if head_kind == "hier":
        m.head.extend(n_new, parent=parent, names=names)
    elif head_kind == "moe":
        m.head.extend(n_new, add_expert=True, freeze_old=False)
    else:
        m.head.extend(n_new)
    if freeze_backbone:
        for p in m.backbone.parameters():
            p.requires_grad_(False)
    rng = np.random.default_rng(seed)
    k = int(round(replay_frac * len(cell.ytr)))
    ridx = rng.choice(len(cell.ytr), size=k, replace=False) if k else np.array([], dtype=int)
    Xf = torch.cat([cell.Xtr[ridx], new_X]) if k else new_X
    yf = np.concatenate([cell.ytr[ridx], n_old + new_y_local]) if k else n_old + new_y_local
    df = np.concatenate([cell.dtr[ridx], np.zeros(len(new_y_local), dtype=int)]) if k else np.zeros(len(new_y_local), dtype=int)
    # a frozen backbone must also keep its BatchNorm statistics fixed (v1 bug: they kept drifting on new-class data)
    cfg = train.TrainConfig(method="erm", epochs=epochs, seed=seed, device=device, lr=1e-3, patience=epochs,
                            freeze_bn=freeze_backbone if freeze_bn is None else freeze_bn)
    # selection set: old-class validation windows plus held-out new-class windows (both from SOURCE sites)
    Xv = torch.cat([cell.Xva, new_val_X])
    yv = np.concatenate([cell.yva, n_old + new_val_y_local])
    m.train()
    m, log = train.fit(m, Xf, yf, df, Xv, yv, n_old + n_new, cfg, n_domains=max(1, int(df.max()) + 1))
    return m, log


def inner_cells(X, meta: pd.DataFrame, split: Split, level: str, seed: int = 0, classes: list[str] | None = None) -> list[Cell]:
    """Leave-one-SOURCE-site-out cells built from the source windows only: the target site never appears.

    Used for model selection, because validation windows from the training sites (v1) overestimate accuracy at a new
    site (v1: 0.80 on validation against 0.72 at the unseen site, correlation 0.30 across runs)."""
    from .splits import leave_one_location_out

    src = np.concatenate([split.train_idx, split.val_idx])
    msrc = meta.iloc[src].reset_index(drop=True)
    Xsrc = X[src]
    return [make_cell(Xsrc, msrc, sp, level, classes=classes) for sp in leave_one_location_out(msrc, seed=seed)]


def select_config(X, meta: pd.DataFrame, split: Split, level: str, head: str, method: str, grid: list[dict], seed: int, epochs: int,
                  device: str = "cpu", site_profiles: dict | None = None, **fixed) -> tuple[int, list[float]]:
    """Index of the grid entry with the best mean balanced accuracy over the inner leave-one-source-site-out cells."""
    scores = []
    for kw in grid:
        accs = []
        for ic in inner_cells(X, meta, split, level, seed=seed):
            n = len(LEVELS[level])
            _, pred, _ = run_method(ic, level, head, method, seed, epochs, device, profiles=site_profiles if kw.get("floor_transplant") or fixed.get("floor_transplant") else None, **fixed, **kw)
            accs.append(balanced_accuracy(ic.yte, pred, n))
        scores.append(float(np.mean(accs)))
    return int(np.argmax(scores)), scores
