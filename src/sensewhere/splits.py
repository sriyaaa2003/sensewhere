"""Leave-one-location-out splits with a subject-disjoint validation set drawn from the source locations only.

The target location is never used for model selection: validation subjects come from the source sites.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    target: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    target_idx: np.ndarray


def leave_one_location_out(
    meta: pd.DataFrame,
    location_col: str = "location",
    subject_col: str = "subject",
    val_fraction: float = 0.15,
    seed: int = 0,
) -> list[Split]:
    """One split per location. Validation is a subject-disjoint slice of the source data, taken per source location.

    Raises if a subject appears in more than one location (that would leak subjects across the domain boundary).
    """
    per_subject_locs = meta.groupby(subject_col)[location_col].nunique()
    if (per_subject_locs > 1).any():
        bad = per_subject_locs[per_subject_locs > 1].index.tolist()[:5]
        raise ValueError(f"subjects appear in several locations: {bad}")
    rng = np.random.default_rng(seed)
    splits = []
    for target in sorted(meta[location_col].unique()):
        is_target = (meta[location_col] == target).to_numpy()
        source = meta[~is_target]
        val_subjects: list = []
        for _, g in source.groupby(location_col):
            subs = np.array(sorted(g[subject_col].unique()))
            k = max(1, int(round(len(subs) * val_fraction)))
            val_subjects.extend(rng.choice(subs, size=k, replace=False).tolist())
        is_val = (~is_target) & meta[subject_col].isin(val_subjects).to_numpy()
        is_train = (~is_target) & (~is_val)
        splits.append(Split(target, np.flatnonzero(is_train), np.flatnonzero(is_val), np.flatnonzero(is_target)))
    return splits


def assert_disjoint(split: Split, meta: pd.DataFrame, subject_col: str = "subject") -> None:
    def subs(idx: np.ndarray) -> set:
        return set(meta.iloc[idx][subject_col])

    tr, va, ta = subs(split.train_idx), subs(split.val_idx), subs(split.target_idx)
    if tr & va or tr & ta or va & ta:
        raise AssertionError("subject overlap between train, validation and target")


def leave_one_domain_out_shared(
    meta: pd.DataFrame,
    domain_col: str,
    group_col: str,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> list[Split]:
    """Leave-one-domain-out where the same groups (e.g. users) appear in every domain, as in Widar torso locations.

    Validation is a group-disjoint slice of the SOURCE domains: its groups never appear in training. The target domain
    contains every group, so these splits measure movement to a new location, not to a new person.
    """
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(meta[group_col].unique()))
    out = []
    for target in sorted(meta[domain_col].unique()):
        is_target = (meta[domain_col] == target).to_numpy()
        k = max(1, int(round(len(groups) * val_fraction)))
        val_groups = rng.choice(groups, size=k, replace=False)
        is_val = (~is_target) & meta[group_col].isin(val_groups).to_numpy()
        is_train = (~is_target) & (~is_val)
        out.append(Split(str(target), np.flatnonzero(is_train), np.flatnonzero(is_val), np.flatnonzero(is_target)))
    return out
