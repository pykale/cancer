"""Tests for patient-level splitting and leakage prevention."""

from __future__ import annotations

import pytest

from kalecancer.loaddata import SplitError, split_patients, stratified_patient_folds
from kalecancer.loaddata.cohort import PatientBag
from kalecancer.loaddata.wsi_feature_access import SlideRecord


def make_bags(num_patients: int = 60, slides_per_patient: int = 2) -> list[PatientBag]:
    """Patients with several slides each and a realistic event rate."""
    bags = []
    for index in range(num_patients):
        patient_id = f"{index:03d}"
        slides = tuple(
            SlideRecord(
                patient_id=patient_id, slide_id=f"PrimaryTumor_HE_{patient_id}_{s}", path=f"/tmp/{patient_id}_{s}.h5"
            )
            for s in range(slides_per_patient)
        )
        bags.append(PatientBag(patient_id=patient_id, slides=slides, duration=100.0 + index, event=int(index % 3 == 0)))
    return bags


def split_patient_ids(split) -> dict[str, set[str]]:
    return {
        name: {bag.patient_id for bag in bags}
        for name, bags in (("train", split.train), ("val", split.val), ("test", split.test))
    }


def test_no_patient_appears_in_two_splits() -> None:
    ids = split_patient_ids(split_patients(make_bags(), seed=1))

    assert ids["train"] & ids["val"] == set()
    assert ids["train"] & ids["test"] == set()
    assert ids["val"] & ids["test"] == set()


def test_every_patient_is_assigned_exactly_once() -> None:
    bags = make_bags()
    ids = split_patient_ids(split_patients(bags, seed=1))

    assert ids["train"] | ids["val"] | ids["test"] == {bag.patient_id for bag in bags}
    assert sum(len(group) for group in ids.values()) == len(bags)


def test_no_slide_appears_in_two_splits() -> None:
    """A patient's slides move together, so slide-level leakage cannot occur."""
    split = split_patients(make_bags(), seed=1)
    slide_ids = {
        name: {slide.slide_id for bag in bags for slide in bag.slides}
        for name, bags in (("train", split.train), ("val", split.val), ("test", split.test))
    }

    assert slide_ids["train"] & slide_ids["val"] == set()
    assert slide_ids["train"] & slide_ids["test"] == set()
    assert slide_ids["val"] & slide_ids["test"] == set()


def test_splitting_is_deterministic_for_a_given_seed() -> None:
    assert split_patient_ids(split_patients(make_bags(), seed=42)) == split_patient_ids(
        split_patients(make_bags(), seed=42)
    )


def test_different_seeds_give_different_splits() -> None:
    assert split_patient_ids(split_patients(make_bags(), seed=1)) != split_patient_ids(
        split_patients(make_bags(), seed=2)
    )


def test_split_sizes_follow_the_requested_ratios() -> None:
    split = split_patients(make_bags(100), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=1)

    assert split.sizes() == {"train": 60, "val": 20, "test": 20}


def test_event_rate_is_preserved_across_splits() -> None:
    split = split_patients(make_bags(300), seed=1)
    rates = [sum(bag.event for bag in bags) / len(bags) for bags in (split.train, split.val, split.test)]

    assert max(rates) - min(rates) < 0.05


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(SplitError, match="must sum to 1.0"):
        split_patients(make_bags(), train_ratio=0.8, val_ratio=0.3, test_ratio=0.2)


def test_too_few_patients_is_rejected() -> None:
    with pytest.raises(SplitError, match="at least 3 patients"):
        split_patients(make_bags(2))


def test_folds_have_disjoint_test_partitions_covering_every_patient() -> None:
    bags = make_bags(60)
    folds = stratified_patient_folds(bags, num_folds=5, seed=1)

    test_sets = [{bag.patient_id for bag in fold.test} for fold in folds]
    assert sum(len(group) for group in test_sets) == len(bags)
    assert set().union(*test_sets) == {bag.patient_id for bag in bags}


def test_each_fold_keeps_train_val_test_disjoint() -> None:
    for fold in stratified_patient_folds(make_bags(60), num_folds=5, seed=1):
        ids = split_patient_ids(fold)
        assert ids["train"] & ids["val"] == set()
        assert ids["train"] & ids["test"] == set()
        assert ids["val"] & ids["test"] == set()


def test_folds_are_deterministic_for_a_given_seed() -> None:
    first = [split_patient_ids(fold) for fold in stratified_patient_folds(make_bags(), num_folds=3, seed=5)]
    second = [split_patient_ids(fold) for fold in stratified_patient_folds(make_bags(), num_folds=3, seed=5)]

    assert first == second


def test_too_few_patients_for_the_requested_folds_is_rejected() -> None:
    with pytest.raises(SplitError, match="at least 5 patients"):
        stratified_patient_folds(make_bags(4), num_folds=5)
