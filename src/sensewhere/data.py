"""Cached-window loading, label maps and the class hierarchy.

The hierarchy is the dataset's own: a deterministic roll-up from the fine (atomic) activity to the motion class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# fine activity -> motion class (from the DopplerWild README; deterministic)
ATOMIC_TO_LOAD = {
    "normal_walking": "normal_walking",
    "hands_in_pockets": "Dual-hand Load",
    "two_hands_carrying": "Dual-hand Load",
    "two_hands_folded": "Dual-hand Load",
    "two_hands_texting": "Dual-hand Load",
    "one_hand_carry_one_hand_pocket": "Dual-hand Load",
    "one_hand_carry_phone_one_hand_carry": "Dual-hand Load",
    "one_hand_fold_one_hand_in_pocket": "Dual-hand Load",
    "one_hand_phone_one_hand_pocket": "Dual-hand Load",
    "cyclist": "On Wheels",
    "on_scooter": "On Wheels",
    "on_a_vehicle": "On Wheels",
}
SINGLE_ATOMIC = [
    "one_hand_carry",
    "one_hand_carry_phone",
    "one_hand_fold",
    "one_hand_in_pocket",
    "one_hand_lift",
    "one_hand_phone_call",
    "shoulder_bag_fold_walking",
]
LOAD_CLASSES = ["normal_walking", "Dual-hand Load", "On Wheels"]


@dataclass(frozen=True)
class Hierarchy:
    tree: dict[str, list[str]]
    leaves: list[str]


def base_hierarchy() -> Hierarchy:
    """root -> {normal_walking, dual, wheels}; dual -> its atomic leaves; wheels -> its atomic leaves."""
    dual = [a for a, l in ATOMIC_TO_LOAD.items() if l == "Dual-hand Load"]
    wheels = [a for a, l in ATOMIC_TO_LOAD.items() if l == "On Wheels"]
    tree = {"root": ["normal_walking", "dual", "wheels"], "dual": dual, "wheels": wheels}
    leaves = ["normal_walking"] + dual + wheels
    return Hierarchy(tree, leaves)


def load_cache(windows_path: str, meta_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(windows_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    if len(meta) != len(X):
        raise ValueError(f"meta rows {len(meta)} != windows {len(X)}")
    return X, meta


def check_hierarchy_consistency(meta: pd.DataFrame) -> None:
    """The stored rolled-up label must equal the roll-up of the fine label (catches silent label corruption)."""
    motion = meta[meta.task == "motion"]
    expect = motion.label_atomic.map(ATOMIC_TO_LOAD)
    bad = motion[expect.to_numpy() != motion.label_load.to_numpy()]
    if len(bad):
        raise ValueError(f"{len(bad)} windows whose rolled-up label disagrees with the hierarchy")


def to_tensor(X: np.ndarray, idx: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(N, F, T) windows get a channel axis; (N, C, H, W) windows (e.g. Widar frames as channels) are used as they are."""
    t = torch.from_numpy(np.asarray(X[idx], dtype=np.float32))
    return (t.unsqueeze(1) if t.ndim == 3 else t).to(device)


# Two alternative trees over the three motion classes (leaf order = LOAD_CLASSES): which split is made first changes
# which confusion each local classifier has to resolve.
HIER_FOOT_FIRST = Hierarchy(
    {"root": ["foot", "On Wheels"], "foot": ["normal_walking", "Dual-hand Load"]},
    ["normal_walking", "Dual-hand Load", "On Wheels"],
)
HIER_WALK_FIRST = Hierarchy(
    {"root": ["normal_walking", "other"], "other": ["Dual-hand Load", "On Wheels"]},
    ["normal_walking", "Dual-hand Load", "On Wheels"],
)
