"""Tests for the ``BaseDataset`` lazy-loading contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kalecancer.loaddata.dataset import BaseDataset


class RecordingDataset(BaseDataset):
    """Minimal concrete subclass that counts how often ``_loader`` runs.

    ``fail_times`` makes the first N loads raise, which is how the retry behaviour is exercised.
    """

    def __init__(self, path=None, identifiers=("a", "b", "c"), fail_times=0):
        super().__init__(path)
        self._seed = list(identifiers)
        self._fail_times = fail_times
        self.loader_calls = 0

    def _loader(self) -> None:
        self.loader_calls += 1
        if self.loader_calls <= self._fail_times:
            raise OSError("source unavailable")
        self.identifiers = list(self._seed)

    def get_by_id(self, identifier):
        return f"sample:{identifier}"


def test_nothing_is_read_until_the_data_is_touched():
    dataset = RecordingDataset()
    assert dataset.loader_calls == 0

    assert len(dataset) == 3
    assert dataset.loader_calls == 1


def test_loader_runs_once_however_often_the_data_is_touched():
    dataset = RecordingDataset()

    assert len(dataset) == 3
    assert dataset[0] == "sample:a"
    assert dataset[2] == "sample:c"
    dataset.load_data()

    assert dataset.loader_calls == 1


def test_getitem_takes_a_position_and_hands_get_by_id_an_identifier():
    assert RecordingDataset(identifiers=("x", "y"))[1] == "sample:y"


def test_a_failed_load_is_retried_rather_than_cached():
    dataset = RecordingDataset(fail_times=1)

    with pytest.raises(OSError):
        dataset.load_data()
    assert dataset.loader_calls == 1

    # The failure left _loaded false, so the next access gets a fresh attempt.
    assert len(dataset) == 3
    assert dataset.loader_calls == 2


@pytest.mark.parametrize(
    "given, expected",
    [("cohort.csv", Path("cohort.csv")), (Path("cohort.csv"), Path("cohort.csv")), (None, None)],
    ids=["str", "path", "none"],
)
def test_path_is_normalised_and_may_be_absent(given, expected):
    assert RecordingDataset(path=given).path == expected


@pytest.mark.parametrize(
    "implemented",
    [{"_loader": lambda self: None}, {"get_by_id": lambda self, identifier: None}, {}],
    ids=["no_get_by_id", "no_loader", "neither"],
)
def test_both_hooks_must_be_implemented(implemented):
    incomplete = type("Incomplete", (BaseDataset,), implemented)

    with pytest.raises(TypeError, match="abstract"):
        incomplete()


def test_composes_with_torch_dataset_utilities():
    dataset = RecordingDataset(identifiers=("a", "b", "c", "d"))

    assert list(Subset(dataset, [0, 2])) == ["sample:a", "sample:c"]
    assert len(ConcatDataset([dataset, dataset])) == 8
    assert list(DataLoader(dataset, batch_size=2)) == [["sample:a", "sample:b"], ["sample:c", "sample:d"]]
