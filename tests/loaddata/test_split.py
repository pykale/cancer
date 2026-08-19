"""Tests for leakage-safe splitting of a metadata table.

The splitter is domain-independent, so the fixtures here are generic tables rather
than pathology cohorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalecancer.loaddata.split import SplitError, composite_labels, k_fold_splits, train_val_test_split

SPLIT_NAMES = ("train", "val", "test")


def make_table(num_groups: int = 60, samples_per_group: int = 2, num_sites: int = 3) -> pd.DataFrame:
    """A table whose rows repeat per group, as when a subject contributes several samples."""
    rows = []
    for index in range(num_groups):
        for sample in range(samples_per_group):
            rows.append(
                {
                    "subject_id": f"s{index:03d}",
                    "sample_id": f"s{index:03d}_{sample}",
                    "label": int(index % 3 == 0),
                    "site": f"site{index % num_sites}",
                }
            )
    return pd.DataFrame(rows)


def groups_of(table: pd.DataFrame, split: dict, key: str = "subject_id") -> dict[str, set]:
    return {name: set(table.iloc[indices][key]) for name, indices in split.items()}


def test_split_returns_indices_into_the_table() -> None:
    table = make_table()

    split = train_val_test_split(table)

    assert set(split) == set(SPLIT_NAMES)
    for indices in split.values():
        assert indices.dtype.kind == "i"
        assert indices.max() < len(table)


def test_every_row_is_assigned_exactly_once() -> None:
    table = make_table()

    split = train_val_test_split(table)

    assigned = np.concatenate(list(split.values()))
    assert sorted(assigned) == list(range(len(table)))


def test_grouped_split_keeps_a_group_in_one_set() -> None:
    table = make_table()

    groups = groups_of(table, train_val_test_split(table, group_key="subject_id"))

    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()


def test_without_a_group_key_rows_split_independently() -> None:
    """Ungrouped splitting is still valid; it simply makes no leakage guarantee."""
    table = make_table()

    split = train_val_test_split(table)

    assert sum(len(indices) for indices in split.values()) == len(table)


def test_stratification_preserves_label_balance() -> None:
    table = make_table(num_groups=300)

    split = train_val_test_split(table, group_key="subject_id", stratify_keys=["label"])
    rates = [table.iloc[indices].drop_duplicates("subject_id")["label"].mean() for indices in split.values()]

    assert max(rates) - min(rates) < 0.1


def test_several_stratify_keys_are_combined_into_one_target() -> None:
    table = make_table(num_groups=180)

    split = train_val_test_split(table, group_key="subject_id", stratify_keys=["label", "site"])

    for indices in split.values():
        assert table.iloc[indices]["site"].nunique() > 1


def test_composite_labels_encode_each_combination() -> None:
    table = pd.DataFrame({"a": [0, 0, 1, 1] * 3, "b": ["x", "y", "x", "y"] * 3})

    labels = composite_labels(table, ["a", "b"], min_count=1)

    assert len(set(labels)) == 4


def test_composite_labels_are_constant_without_keys() -> None:
    assert set(composite_labels(make_table(), [])) == {0}


def test_rare_combinations_are_absorbed_rather_than_failing() -> None:
    """A class too small for every split joins the majority instead of raising."""
    table = pd.DataFrame({"label": [0] * 20 + [1]})

    labels = composite_labels(table, ["label"], min_count=5)

    assert len(set(labels)) == 1


def test_every_retained_class_meets_the_minimum_count() -> None:
    table = pd.DataFrame({"label": [0] * 20 + [1] * 2 + [2] * 3})

    counts = pd.Series(composite_labels(table, ["label"], min_count=5)).value_counts()

    assert counts.min() >= 5


def test_split_succeeds_when_a_stratum_is_too_rare() -> None:
    table = make_table(num_groups=40)
    table.loc[table.index[:2], "label"] = 7

    split = train_val_test_split(table, group_key="subject_id", stratify_keys=["label"])

    assert all(len(indices) > 0 for indices in split.values())


def test_splitting_is_deterministic_for_a_given_seed() -> None:
    table = make_table()

    first = train_val_test_split(table, group_key="subject_id", seed=7)
    second = train_val_test_split(table, group_key="subject_id", seed=7)

    assert all(np.array_equal(first[name], second[name]) for name in SPLIT_NAMES)


def test_different_seeds_give_different_splits() -> None:
    table = make_table()

    first = train_val_test_split(table, group_key="subject_id", seed=1)
    second = train_val_test_split(table, group_key="subject_id", seed=2)

    assert not all(np.array_equal(first[name], second[name]) for name in SPLIT_NAMES)


def test_ratios_control_the_holdout_sizes() -> None:
    table = make_table(num_groups=200, samples_per_group=1)

    split = train_val_test_split(table, val_ratio=0.2, test_ratio=0.2, group_key="subject_id")

    assert 0.1 < len(split["test"]) / len(table) < 0.3
    assert len(split["train"]) > len(split["val"])


@pytest.mark.parametrize("ratios", [(0.0, 0.0), (0.6, 0.6)])
def test_invalid_ratios_are_rejected(ratios: tuple[float, float]) -> None:
    with pytest.raises(SplitError, match="must lie in"):
        train_val_test_split(make_table(), val_ratio=ratios[0], test_ratio=ratios[1])


def test_unknown_group_key_is_rejected() -> None:
    with pytest.raises(SplitError, match="group_key 'missing' is not a column"):
        train_val_test_split(make_table(), group_key="missing")


def test_unknown_stratify_key_is_rejected() -> None:
    with pytest.raises(SplitError, match="stratify_keys"):
        train_val_test_split(make_table(), stratify_keys=["missing"])


def test_too_few_groups_is_rejected() -> None:
    with pytest.raises(SplitError, match="at least 3 unique subject_id"):
        train_val_test_split(make_table(num_groups=2), group_key="subject_id")


def test_folds_cover_every_group_exactly_once() -> None:
    table = make_table()

    folds = k_fold_splits(table, num_folds=5, group_key="subject_id")
    test_groups = [set(table.iloc[fold["test"]]["subject_id"]) for fold in folds]

    assert set().union(*test_groups) == set(table["subject_id"])
    assert sum(len(groups) for groups in test_groups) == table["subject_id"].nunique()


def test_each_fold_keeps_the_three_sets_disjoint() -> None:
    table = make_table()

    for fold in k_fold_splits(table, num_folds=5, group_key="subject_id"):
        groups = groups_of(table, fold)
        assert groups["train"] & groups["val"] == set()
        assert groups["train"] & groups["test"] == set()
        assert groups["val"] & groups["test"] == set()


def test_folds_are_deterministic_for_a_given_seed() -> None:
    table = make_table()

    first = k_fold_splits(table, num_folds=3, group_key="subject_id", seed=5)
    second = k_fold_splits(table, num_folds=3, group_key="subject_id", seed=5)

    assert all(np.array_equal(a["test"], b["test"]) for a, b in zip(first, second, strict=True))


def test_folds_work_without_grouping() -> None:
    table = make_table(samples_per_group=1)

    folds = k_fold_splits(table, num_folds=4, stratify_keys=["label"])

    assert len(folds) == 4


def test_too_few_folds_is_rejected() -> None:
    with pytest.raises(SplitError, match="at least 2"):
        k_fold_splits(make_table(), num_folds=1)


def test_too_few_groups_for_the_requested_folds_is_rejected() -> None:
    with pytest.raises(SplitError, match="at least 5 unique subject_id"):
        k_fold_splits(make_table(num_groups=4), num_folds=5, group_key="subject_id")
