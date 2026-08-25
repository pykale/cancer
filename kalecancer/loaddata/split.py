"""Leakage-safe dataset splitting over a metadata table.

Splits are described by column names rather than by domain concepts, so the same
utility serves slides grouped by patient, cardiac volumes grouped by subject, or
plain tabular rows:

    >>> splits = train_val_test_split(metadata, group_key="patient_id", stratify_keys=["event"])

Samples sharing a ``group_key`` value always land in the same split, which is what
prevents leakage when one subject contributes several samples. Stratification keys
are combined into a single categorical target, so one, several, or no keys are
handled identically.

Splitters return positional indices into the table; callers select rows with
``metadata.iloc[...]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

Split = dict[str, np.ndarray]

#: Upper bound on the folds a holdout is built from, so the share it can express is
#: no coarser than one part in this many.
MAX_HOLDOUT_FOLDS = 20


class SplitError(ValueError):
    """Raised when a requested split cannot be produced."""


def composite_labels(metadata: pd.DataFrame, stratify_keys: Sequence[str] = (), min_count: int = 2) -> np.ndarray:
    """Combine stratification columns into a single categorical target.

    A class with fewer members than the number of splits cannot appear in all of
    them, so rare combinations are absorbed into the most common class. Stratification
    then degrades gracefully instead of failing on an unlucky combination.

    Args:
        metadata: Table describing the samples.
        stratify_keys: Columns to stratify on. Empty gives a constant target, which
            reduces stratified splitting to plain shuffling.
        min_count: Smallest class size to keep separate.

    Returns:
        Integer labels, one per row.

    Raises:
        SplitError: If a requested column is missing.
    """
    missing = [key for key in stratify_keys if key not in metadata.columns]
    if missing:
        raise SplitError(f"stratify_keys {missing} are not columns of the metadata: {list(metadata.columns)}")

    if not stratify_keys or metadata.empty:
        return np.zeros(len(metadata), dtype=np.int64)

    combined = metadata[list(stratify_keys)].astype(str).agg("|".join, axis=1)
    labels, _ = pd.factorize(combined)
    counts = pd.Series(labels).value_counts()
    return np.where(np.isin(labels, counts[counts >= min_count].index), labels, counts.idxmax())


def _holdout(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    ratio: float,
    group_key: str | None,
    stratify_keys: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out approximately ``ratio`` of ``indices``, keeping groups intact.

    Built from a K-fold splitter so grouped and ungrouped splitting stay on one code
    path: enough whole folds are merged to reach the requested share. The achievable
    resolution is one fold, so a small table -- which supports few folds -- gives a
    coarser share than a large one.

    Raises:
        SplitError: If fewer than two independent units are available to split.
    """
    subset = metadata.iloc[indices]
    units = subset[group_key].nunique() if group_key else len(subset)
    if units < 2:
        unit = f"unique {group_key}" if group_key else "rows"
        raise SplitError(f"need at least 2 {unit} to hold part out, got {units}")

    # More folds give a finer share; never more than the data can support.
    num_splits = max(2, min(MAX_HOLDOUT_FOLDS, units))
    labels = composite_labels(subset, stratify_keys, min_count=num_splits)
    groups = subset[group_key].to_numpy() if group_key else None

    splitter = StratifiedGroupKFold if group_key else StratifiedKFold
    folds = list(splitter(n_splits=num_splits, shuffle=True, random_state=seed).split(subset, labels, groups))

    # Folds vary in size, so the count is chosen by how many rows it actually holds
    # out rather than by fold arithmetic. At least one fold is held out, and at least
    # one is kept, whatever the ratio.
    sizes = np.cumsum([len(folds[fold][1]) for fold in range(num_splits - 1)])
    taken = int(np.argmin(np.abs(sizes - ratio * len(subset)))) + 1
    held = np.concatenate([folds[fold][1] for fold in range(taken)])
    keep = np.setdiff1d(np.arange(len(subset)), held)
    return indices[keep], indices[np.sort(held)]


def _validate(metadata: pd.DataFrame, group_key: str | None, minimum: int) -> None:
    if group_key is not None and group_key not in metadata.columns:
        raise SplitError(f"group_key {group_key!r} is not a column of the metadata: {list(metadata.columns)}")
    units = metadata[group_key].nunique() if group_key else len(metadata)
    if units < minimum:
        unit = f"unique {group_key}" if group_key else "rows"
        raise SplitError(f"need at least {minimum} {unit} to split, got {units}")


def holdout_split(
    metadata: pd.DataFrame,
    ratio: float = 0.15,
    group_key: str | None = None,
    stratify_keys: Sequence[str] = (),
    seed: int = 2026,
) -> Split:
    """Split a metadata table in two.

    For when a third set is not wanted: carving a validation set out of a fixed
    training half, for instance, where the test set is already decided.

    Args:
        metadata: Table describing the samples.
        ratio: Approximate share held out.
        group_key: Column whose values must not be split across sets.
        stratify_keys: Columns whose distribution should be preserved.
        seed: Seed making the split reproducible.

    Returns:
        Positional indices keyed by ``"fit"`` and ``"holdout"``.

    Raises:
        SplitError: If the ratio is invalid or the table is too small to split.
    """
    if not 0 < ratio < 1:
        raise SplitError(f"ratio must lie in (0, 1), got {ratio}")
    _validate(metadata, group_key, minimum=2)

    fit, holdout = _holdout(metadata, np.arange(len(metadata)), ratio, group_key, stratify_keys, seed)
    return {"fit": fit, "holdout": holdout}


def train_val_test_split(
    metadata: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_key: str | None = None,
    stratify_keys: Sequence[str] = (),
    seed: int = 2026,
) -> Split:
    """Split a metadata table into train, validation and test indices.

    Args:
        metadata: Table describing the samples.
        val_ratio: Approximate share held out for validation.
        test_ratio: Approximate share held out for testing.
        group_key: Column whose values must not be split across sets.
        stratify_keys: Columns whose distribution should be preserved.
        seed: Seed making the split reproducible.

    Returns:
        Positional indices keyed by ``"train"``, ``"val"`` and ``"test"``.

    Raises:
        SplitError: If the ratios are invalid or the table is too small to split.
    """
    if not 0 < val_ratio + test_ratio < 1:
        raise SplitError(f"val_ratio + test_ratio must lie in (0, 1), got {val_ratio + test_ratio}")
    _validate(metadata, group_key, minimum=3)

    all_indices = np.arange(len(metadata))
    remaining, test = _holdout(metadata, all_indices, test_ratio, group_key, stratify_keys, seed)
    train, val = _holdout(metadata, remaining, val_ratio / (1 - test_ratio), group_key, stratify_keys, seed)
    return {"train": train, "val": val, "test": test}


def k_fold_splits(
    metadata: pd.DataFrame,
    num_folds: int = 5,
    val_ratio: float = 0.15,
    group_key: str | None = None,
    stratify_keys: Sequence[str] = (),
    seed: int = 2026,
) -> list[Split]:
    """Build cross-validation folds over a metadata table.

    Each fold holds out one test partition; validation is carved from that fold's
    remaining samples, so the three sets stay disjoint.

    Args:
        metadata: Table describing the samples.
        num_folds: Number of folds.
        val_ratio: Share of each fold's non-test samples held out for validation.
        group_key: Column whose values must not be split across sets.
        stratify_keys: Columns whose distribution should be preserved.
        seed: Seed making the folds reproducible.

    Returns:
        One index mapping per fold.

    Raises:
        SplitError: If ``num_folds`` is below 2 or the table is too small.
    """
    if num_folds < 2:
        raise SplitError(f"num_folds must be at least 2, got {num_folds}")
    _validate(metadata, group_key, minimum=num_folds)

    labels = composite_labels(metadata, stratify_keys, min_count=num_folds)
    groups = metadata[group_key].to_numpy() if group_key else None
    splitter = StratifiedGroupKFold if group_key else StratifiedKFold
    folds = splitter(n_splits=num_folds, shuffle=True, random_state=seed)

    splits = []
    for fold, (fit, test) in enumerate(folds.split(metadata, labels, groups)):
        train, val = _holdout(metadata, fit, val_ratio, group_key, stratify_keys, seed + fold)
        splits.append({"train": train, "val": val, "test": test})
    return splits
