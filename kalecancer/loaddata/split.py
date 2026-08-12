"""Leakage-safe, patient-level dataset splitting.

Splits are computed over patients, never over slides or patches. Because a
:class:`~kalecancer.loaddata.cohort.PatientBag` already pools all of a patient's
slides, no slide or patch can appear in two splits. Splits are stratified on the
event indicator so the censoring balance is preserved, and seeded so they are
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold, train_test_split

from kalecancer.loaddata.cohort import PatientBag


class SplitError(ValueError):
    """Raised when a requested split cannot be produced."""


@dataclass(frozen=True)
class CohortSplit:
    """Disjoint patient bags for one experiment."""

    train: list[PatientBag]
    val: list[PatientBag]
    test: list[PatientBag]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _stratify_labels(bags: list[PatientBag]) -> list[int]:
    return [bag.event for bag in bags]


def _can_stratify(labels: list[int], num_groups: int) -> bool:
    """Stratification needs at least ``num_groups`` members in every class."""
    return len(set(labels)) > 1 and all(labels.count(label) >= num_groups for label in set(labels))


def split_patients(
    bags: list[PatientBag],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 2026,
) -> CohortSplit:
    """Split patients into train/validation/test sets.

    Args:
        bags: Matched patient bags.
        train_ratio: Fraction of patients used for training.
        val_ratio: Fraction used for validation (model selection).
        test_ratio: Fraction used for the final held-out evaluation.
        seed: Seed making the split reproducible.

    Returns:
        Disjoint patient-level splits.

    Raises:
        SplitError: If the ratios do not sum to 1, or any split would be empty.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"split ratios must sum to 1.0, got {total}")
    if len(bags) < 3:
        raise SplitError(f"need at least 3 patients to build three splits, got {len(bags)}")

    labels = _stratify_labels(bags)
    train_bags, holdout = train_test_split(
        bags,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels if _can_stratify(labels, 2) else None,
    )

    holdout_labels = _stratify_labels(holdout)
    val_share = val_ratio / (val_ratio + test_ratio)
    val_bags, test_bags = train_test_split(
        holdout,
        train_size=val_share,
        random_state=seed,
        shuffle=True,
        stratify=holdout_labels if _can_stratify(holdout_labels, 2) else None,
    )

    split = CohortSplit(train=train_bags, val=val_bags, test=test_bags)
    empty = [name for name, size in split.sizes().items() if size == 0]
    if empty:
        raise SplitError(f"split(s) {empty} are empty; adjust the ratios for {len(bags)} patients")
    return split


def stratified_patient_folds(
    bags: list[PatientBag],
    num_folds: int = 5,
    val_ratio: float = 0.15,
    seed: int = 2026,
) -> list[CohortSplit]:
    """Build patient-level cross-validation folds.

    Each fold holds out one test partition; the validation set is carved out of that
    fold's training patients, so train, validation and test remain disjoint.

    Args:
        bags: Matched patient bags.
        num_folds: Number of cross-validation folds.
        val_ratio: Fraction of each fold's training patients held out for validation.
        seed: Seed making the folds reproducible.

    Raises:
        SplitError: If fewer patients than folds are available.
    """
    if num_folds < 2:
        raise SplitError(f"num_folds must be at least 2, got {num_folds}")
    if len(bags) < num_folds:
        raise SplitError(f"need at least {num_folds} patients for {num_folds} folds, got {len(bags)}")

    labels = _stratify_labels(bags)
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    strata = labels if _can_stratify(labels, num_folds) else [0] * len(bags)

    folds = []
    for fold, (fit_index, test_index) in enumerate(splitter.split(bags, strata)):
        fit_bags = [bags[i] for i in fit_index]
        test_bags = [bags[i] for i in test_index]

        fit_labels = _stratify_labels(fit_bags)
        train_bags, val_bags = train_test_split(
            fit_bags,
            test_size=val_ratio,
            random_state=seed + fold,
            shuffle=True,
            stratify=fit_labels if _can_stratify(fit_labels, 2) else None,
        )
        folds.append(CohortSplit(train=train_bags, val=val_bags, test=test_bags))
    return folds
