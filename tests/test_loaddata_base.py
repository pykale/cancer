"""Tests for ``kalecancer.loaddata.base``.

``BaseDataset`` is exercised through a deliberately trivial concrete subclass
rather than through ``TabularDataset``. The point is to pin the *base* contract
-- two-phase loading, identifier keying, fit-returns-a-new-object -- independently
of any one modality, because ``WSIDataset`` and ``MultiModalDataset`` are going to
inherit exactly this and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from kalecancer.loaddata.base import BaseDataset, NotFittedError, Target

# --------------------------------------------------------------------------- #
# minimal implementations of the two abstract halves of the contract
# --------------------------------------------------------------------------- #


class ToyDataset(BaseDataset):
    """Index is a whitespace-separated list of identifiers; payload is derived.

    Payload is deliberately *not* read during ``_load_index``, mirroring the
    manifest-then-slides split that motivates the two-phase design.
    """

    def _load_index(self) -> None:
        assert self.path is not None  # BaseDataset only calls this when a path was given
        self.identifiers = self.path.read_text().split()

    def get_by_id(self, identifier) -> torch.Tensor:
        return torch.tensor([float(identifier), float(identifier) * 2], dtype=torch.float32)


class UnfittedToyDataset(ToyDataset):
    """A subclass that never considers itself fitted, to reach the guard."""

    @property
    def is_fitted(self) -> bool:
        return False


class StubTarget:
    """A non-survival target, implementing only what the protocol requires."""

    def bind(self, frame, identifier) -> None:  # pragma: no cover - never called by base
        raise AssertionError("BaseDataset must not call bind(); its owner does")

    def for_(self, identifier) -> dict:
        return {"label": torch.tensor(float(identifier) % 2, dtype=torch.float32)}


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    path = tmp_path / "index.txt"
    path.write_text("10 20 30 40 50")
    return path


@pytest.fixture
def toy(index_path: Path) -> ToyDataset:
    return ToyDataset(path=index_path, name="toy")


# --------------------------------------------------------------------------- #
# the subclass contract
# --------------------------------------------------------------------------- #


def test_base_dataset_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BaseDataset(path=None)


def test_both_halves_of_the_contract_are_abstract():
    assert BaseDataset.__abstractmethods__ == frozenset({"_load_index", "get_by_id"})


def test_is_a_torch_dataset(toy):
    """So DataLoader, Subset, random_split and ConcatDataset work without adapters."""
    assert isinstance(toy, TorchDataset)


# --------------------------------------------------------------------------- #
# index loading
# --------------------------------------------------------------------------- #


def test_index_is_loaded_at_construction(toy):
    assert toy.identifiers == ["10", "20", "30", "40", "50"]
    assert len(toy) == 5


def test_no_path_means_no_index_load(index_path):
    """Composite datasets pass ``path=None`` and take their index from components."""

    class Exploding(ToyDataset):
        def _load_index(self) -> None:  # pragma: no cover - must never run
            raise AssertionError("_load_index must not be called when path is None")

    dataset = Exploding(path=None)
    assert dataset.path is None
    assert dataset.identifiers == []
    assert len(dataset) == 0


def test_a_subclass_can_declare_an_index_source_that_is_not_a_path():
    """The hook that lets ``TabularDataset`` accept an already-loaded frame.

    Gating index loading on ``self.path`` alone would mean a subclass handed its
    data in memory had to re-run ``_load_index`` and ``_reindex`` itself, keeping a
    copy of the base class's construction order in step by hand.
    """

    class InMemory(ToyDataset):
        def __init__(self, identifiers, **kwargs):
            self._source = identifiers
            super().__init__(**kwargs)

        def _has_index_source(self) -> bool:
            return self.path is not None or self._source is not None

        def _load_index(self) -> None:
            self.identifiers = list(self._source)

    dataset = InMemory(["10", "20", "30"])

    assert dataset.path is None
    assert dataset.identifiers == ["10", "20", "30"]
    assert "20" in dataset, "_reindex ran too, so membership works"


def test_path_is_coerced_to_pathlib(index_path):
    assert ToyDataset(path=str(index_path)).path == index_path


def test_defaults(index_path):
    dataset = ToyDataset(path=index_path)
    assert dataset.name == "features"
    assert dataset.target is None
    assert dataset.is_fitted is True, "the base class holds no fitted state"


# --------------------------------------------------------------------------- #
# identifier keying
# --------------------------------------------------------------------------- #


def test_contains_uses_the_identifier_lookup(toy):
    assert "30" in toy
    assert "31" not in toy
    assert 30 not in toy, "identifiers are strings here; no coercion"


def test_row_lookup_matches_identifier_order(toy):
    assert toy._row_of == {"10": 0, "20": 1, "30": 2, "40": 3, "50": 4}


def test_getitem_is_positional_but_payload_is_by_identifier(toy):
    item = toy[2]
    assert item["patient_id"] == "30"
    torch.testing.assert_close(item["toy"], toy.get_by_id("30"))


def test_getitem_supports_negative_indices(toy):
    assert toy[-1]["patient_id"] == "50"


# --------------------------------------------------------------------------- #
# the item dict
# --------------------------------------------------------------------------- #


def test_item_dict_shape_without_a_target(toy):
    assert set(toy[0]) == {"toy", "patient_id"}


def test_target_contributions_are_merged_into_the_item(index_path):
    dataset = ToyDataset(path=index_path, name="toy", target=StubTarget())
    item = dataset[0]
    assert set(item) == {"toy", "patient_id", "label"}
    assert item["label"].item() == 0.0


def test_base_never_calls_bind(index_path):
    """``bind`` belongs to whoever owns the frame; ``StubTarget`` asserts if called."""
    ToyDataset(path=index_path, target=StubTarget())[0]


def test_name_is_the_modality_key(index_path):
    assert "wsi" in ToyDataset(path=index_path, name="wsi")[0]


def test_a_stub_target_satisfies_the_runtime_protocol():
    assert isinstance(StubTarget(), Target)


def test_default_collate_handles_the_item_dict(index_path):
    dataset = ToyDataset(path=index_path, name="toy", target=StubTarget())
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))

    assert batch["toy"].shape == (3, 2)
    assert batch["label"].shape == (3,)
    assert batch["patient_id"] == ["10", "20", "30"], "strings collate to a list"


# --------------------------------------------------------------------------- #
# fold discipline
# --------------------------------------------------------------------------- #


def test_subset_returns_a_new_object_and_leaves_the_original_alone(toy):
    sub = toy.subset([0, 2])

    assert sub is not toy
    assert sub.identifiers == ["10", "30"]
    assert toy.identifiers == ["10", "20", "30", "40", "50"]


def test_subset_reindexes_so_lookups_stay_correct(toy):
    sub = toy.subset([3, 4])

    assert sub._row_of == {"40": 0, "50": 1}
    assert "10" not in sub
    assert sub[0]["patient_id"] == "40"


def test_subset_preserves_the_order_it_is_given(toy):
    """``identifiers`` is the ordering authority, and callers own that order."""
    assert toy.subset([4, 0, 2]).identifiers == ["50", "10", "30"]


def test_subset_carries_configuration(index_path):
    dataset = ToyDataset(path=index_path, name="toy", target=StubTarget())
    sub = dataset.subset([1])
    assert sub.name == "toy"
    assert sub.target is dataset.target


def test_empty_subset_is_valid(toy):
    empty = toy.subset([])
    assert len(empty) == 0
    assert list(empty.identifiers) == []


def test_fit_transform_returns_a_new_instance_when_there_is_nothing_to_fit(toy):
    """Even with no state to fit, the caller's object must come back untouched."""
    fitted = toy.fit_transform()
    assert fitted is not toy
    assert fitted.identifiers == toy.identifiers


def test_restricting_before_fitting_scopes_the_fit_to_those_rows(toy):
    """The fold idiom: ``subset`` owns the positional indices, ``fit_transform`` the state."""
    fitted = toy.subset([1, 2]).fit_transform()
    assert fitted.identifiers == ["20", "30"]
    assert toy.identifiers == ["10", "20", "30", "40", "50"], "the parent is untouched"


def test_transform_passes_other_through_when_fitted(toy):
    other = toy.subset([0])
    assert toy.transform(other) is other


def test_transform_raises_when_unfitted(index_path):
    dataset = UnfittedToyDataset(path=index_path)
    with pytest.raises(NotFittedError, match="fit_transform"):
        dataset.transform(dataset)


def test_not_fitted_error_is_a_runtime_error():
    """So callers already catching RuntimeError are not surprised."""
    assert issubclass(NotFittedError, RuntimeError)


def test_raw_is_a_no_op_on_the_base_class(toy):
    assert toy.raw is toy


# --------------------------------------------------------------------------- #
# payload is never read during indexing
# --------------------------------------------------------------------------- #


def test_payload_is_only_read_on_demand(index_path):
    """The split that earns its keep for WSI: index reads a manifest, not slides."""
    reads: list[str] = []

    class Counting(ToyDataset):
        def get_by_id(self, identifier):
            reads.append(identifier)
            return super().get_by_id(identifier)

    dataset = Counting(path=index_path)
    assert reads == [], "construction must not touch payload"

    dataset[1]
    assert reads == ["20"]
