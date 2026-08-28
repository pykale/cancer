"""The cohort/view split: what the base class guarantees for every modality.

Exercised through toy cohorts rather than ``TabularCohort``, so what is being
tested is the contract itself and not one implementation of it. The toys stand in
for the modality that motivates the design -- a slide cohort, whose payload is
expensive, lazy and *stochastic*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from kalecancer.loaddata.multimodal_access import PatientSample, collate_samples
from kalecancer.loaddata.tabular_access import Cohort, LeakageError, NotFittedError


class ToyCohort(Cohort):
    """One identifier per line; payload derived from the identifier, counted."""

    def __init__(self, path=None, name="features", target=None):
        self.payload_calls: list[str] = []
        super().__init__(path=path, name=name, target=target)

    def _load_index(self) -> None:
        assert self.path is not None  # only reached when a path was given
        self.identifiers = self.path.read_text(encoding="utf-8").split()

    def fit_preprocessor(self, identifiers):
        return None

    def payload(self, identifier, prep):
        self.payload_calls.append(identifier)
        return {self.name: torch.full((3,), float(identifier))}


class BulkToyCohort(ToyCohort):
    """Volunteers an eager path, the way a small table does."""

    def payload_bulk(self, identifiers, prep):
        rows = [torch.full((3,), float(i)) for i in identifiers]
        return {self.name: torch.stack(rows) if rows else torch.zeros((0, 3))}


class StochasticToyCohort(ToyCohort):
    """Payload changes every call, the way per-epoch tile sampling does."""

    def payload(self, identifier, prep):
        self.payload_calls.append(identifier)
        return {self.name: torch.rand(3)}


class MisorderedBulkCohort(ToyCohort):
    """Right number of rows, wrong order -- the case a length check cannot catch."""

    def payload_bulk(self, identifiers, prep):
        rows = [torch.full((3,), float(i)) for i in reversed(list(identifiers))]
        return {self.name: torch.stack(rows)}


class WholeCohortBulkCohort(ToyCohort):
    """Ignores the subset it was asked for and returns everything -- the classic slip."""

    def payload_bulk(self, identifiers, prep):
        return {self.name: torch.stack([torch.full((3,), float(i)) for i in self.identifiers])}


class ShortBulkCohort(ToyCohort):
    """Returns fewer rows than it was asked for."""

    def payload_bulk(self, identifiers, prep):
        return {self.name: torch.zeros((len(identifiers) - 1, 3))}


class NanBulkCohort(ToyCohort):
    """Features legitimately containing NaN, consistently on both paths.

    Reachable in practice: ``continuous_transform=StandardScaler()`` with no
    imputer propagates NaN straight through.
    """

    def payload(self, identifier, prep):
        self.payload_calls.append(identifier)
        return {self.name: torch.tensor([float(identifier), float("nan"), 1.0])}

    def payload_bulk(self, identifiers, prep):
        rows = [torch.tensor([float(i), float("nan"), 1.0]) for i in identifiers]
        return {self.name: torch.stack(rows)}


class StubTarget:
    required_columns = ("outcome",)

    def __init__(self):
        self.bound = None

    def bind(self, identifiers, values):
        self.bound = (list(identifiers), values)

    def for_(self, identifier):
        return {"outcome": torch.tensor(float(identifier))}

    def values_for(self, identifiers):
        return {"outcome": torch.tensor([float(i) for i in identifiers])}

    def stratify_labels(self, identifiers):
        return np.array([int(i) % 2 for i in identifiers])


#: The toy index is "0".."9", so an identifier reads as its own row number.
TOY_IDS = [str(i) for i in range(10)]


@pytest.fixture
def index_path(tmp_path) -> Path:
    path = tmp_path / "index.txt"
    path.write_text("\n".join(str(i) for i in range(10)), encoding="utf-8")
    return path


@pytest.fixture
def toy(index_path) -> ToyCohort:
    return ToyCohort(index_path)


# =========================================================================== #
# the subclass contract
# =========================================================================== #


def test_cohort_is_abstract():
    with pytest.raises(TypeError):
        Cohort()


def test_exactly_three_methods_are_abstract():
    """Kept small on purpose. Every abstract method is a tax on the next modality."""
    assert Cohort.__abstractmethods__ == frozenset({"_load_index", "fit_preprocessor", "payload"})


def test_a_cohort_is_not_a_torch_dataset(toy):
    """A cohort with no preprocessor is not a dataset, so it does not pretend to be.

    ``CohortView`` is the only ``Dataset`` in the package; that is what stops
    anyone iterating a cohort and quietly getting untransformed values.
    """
    assert not isinstance(toy, TorchDataset)
    assert not hasattr(toy, "__getitem__")


def test_a_view_is_a_torch_dataset(toy):
    view = toy.view(TOY_IDS[:4], None)
    assert isinstance(view, TorchDataset)
    assert len(view) == 4


# =========================================================================== #
# index loading -- phase one
# =========================================================================== #


def test_index_is_loaded_at_construction(toy):
    assert toy.identifiers == [str(i) for i in range(10)]
    assert len(toy) == 10


def test_no_source_means_no_index_load(index_path):
    """Composite cohorts take their index from their components, not from a path."""
    assert ToyCohort(path=None).identifiers == []


def test_a_subclass_can_declare_a_source_that_is_not_a_path(index_path):
    class FromMemory(ToyCohort):
        def _has_index_source(self):
            return True

        def _load_index(self):
            self.identifiers = ["a", "b"]

    assert FromMemory(path=None).identifiers == ["a", "b"]


def test_path_is_coerced_to_pathlib(index_path):
    assert ToyCohort(str(index_path)).path == index_path


def test_defaults(index_path):
    cohort = ToyCohort(index_path)
    assert cohort.name == "features"
    assert cohort.target is None


def test_loading_the_index_reads_no_payload(index_path):
    """Rule 3, and the reason a slide cohort can be constructed in milliseconds."""
    cohort = ToyCohort(index_path)
    assert cohort.payload_calls == [], "constructing a cohort must not touch payload"


# =========================================================================== #
# identifier keying
# =========================================================================== #


def test_contains_and_index_of_use_the_identifier_lookup(toy):
    assert "3" in toy
    assert "no-such-sample" not in toy
    np.testing.assert_array_equal(toy.index_of(["3", "0"]), [3, 0])


def test_a_view_reports_its_own_identifiers_in_its_own_order(toy):
    view = toy.view(["7", "1", "4"], None)
    assert view.identifiers == ["7", "1", "4"]


def test_a_view_indexes_into_itself_not_into_the_cohort(toy):
    """``view[0]`` is the view's first row, which is rarely the cohort's."""
    view = toy.view(["7", "1", "4"], None)
    assert view[0].patient_id == "7"
    assert view[2].patient_id == "4"


def test_a_view_supports_negative_indices(toy):
    """``DataLoader`` never uses them, but people do, and silently wrapping to the
    cohort's last row rather than the view's would be a hard bug to see."""
    view = toy.view(["7", "1", "4"], None)
    assert view[-1].patient_id == view[2].patient_id == "4"


def test_an_empty_view_is_valid(toy):
    assert len(toy.view([], None)) == 0


# =========================================================================== #
# payload -- phase two
# =========================================================================== #


def test_payload_is_only_read_on_demand(toy):
    """Building a view opens nothing; asking for an item is what reads."""
    view = toy.view(TOY_IDS[:10], None)
    assert toy.payload_calls == []

    view[3]
    assert toy.payload_calls == ["3"]


def test_a_view_yields_patient_samples(toy):
    item = toy.view(TOY_IDS[:3], None)[1]
    assert isinstance(item, PatientSample)
    assert item.patient_id == "1"
    assert item.modalities["features"].shape == (3,)
    assert item.present["features"].item() is True
    assert item.target == {}, "no target means no supervision keys"


def test_the_name_is_the_modality_key(index_path):
    view = ToyCohort(index_path, name="covariates").view(["0"], None)
    assert set(view[0].modalities) == {"covariates"}


def test_a_target_contributes_its_own_keys(index_path):
    view = ToyCohort(index_path, target=StubTarget()).view(["5"], None)
    assert view[0].target == {"outcome": torch.tensor(5.0)}


def test_a_target_is_checked_at_construction(index_path):
    """A mis-shaped target must fail here, not three hours into training."""
    with pytest.raises(TypeError, match="not a valid Target"):
        ToyCohort(index_path, target=object())


def test_the_base_class_never_binds(index_path):
    """Binding needs values, which only a concrete cohort knows how to produce."""
    target = StubTarget()
    ToyCohort(index_path, target=target)
    assert target.bound is None


# =========================================================================== #
# caching -- opt-in, and off for stochastic payloads
# =========================================================================== #


def test_a_cohort_that_volunteers_a_bulk_path_is_cached(index_path):
    cohort = BulkToyCohort(index_path)
    view = cohort.view(TOY_IDS[:4], None)

    assert cohort.payload_calls == ["0"], "one call, the alignment spot check"
    values = [view[i].modalities["features"] for i in range(4)]
    assert cohort.payload_calls == ["0"], "iterating must not reach the per-sample path"
    torch.testing.assert_close(values[2], torch.full((3,), 2.0))


def test_a_stochastic_payload_is_never_cached(index_path):
    """The guarantee a slide cohort depends on.

    Tile sampling is per-epoch, so caching would freeze one draw for a whole run and
    look like it was working. Caching is opt-in via ``payload_bulk``.
    """
    cohort = StochasticToyCohort(index_path)
    view = cohort.view(TOY_IDS[:4], None)

    first = view[0].modalities["features"]
    second = view[0].modalities["features"]

    assert not torch.equal(first, second), "the same row must be re-read, not replayed"
    assert cohort.payload_calls == ["0", "0"]


def test_a_bulk_block_in_the_wrong_order_is_caught(index_path):
    """Rows are read positionally while identifiers come from the index.

    A cohort returning its own order pairs every patient with someone else's
    features -- it trains perfectly and means nothing. This is the one door in the
    design that has to be positional, so it is the one that gets checked.
    """
    cohort = MisorderedBulkCohort(index_path)
    with pytest.raises(ValueError, match="disagrees with payload"):
        cohort.view(["2", "3", "5"], None)


def test_a_bulk_block_of_the_wrong_length_is_caught(index_path):
    """The cheap half of the check: asked for a subset, handed the whole cohort."""
    with pytest.raises(ValueError, match="returned 10 rows .* asked for 3"):
        WholeCohortBulkCohort(index_path).view(["2", "3", "5"], None)

    with pytest.raises(ValueError, match="returned 3 rows .* asked for 4"):
        ShortBulkCohort(index_path).view(TOY_IDS[:4], None)


def test_a_correctly_ordered_bulk_block_passes(index_path):
    """The check must not fire on the implementation it is meant to allow."""
    view = BulkToyCohort(index_path).view(["7", "1", "4"], None)
    assert [v.modalities["features"][0].item() for v in (view[0], view[1], view[2])] == [7.0, 1.0, 4.0]


def test_nan_features_do_not_trip_the_alignment_check(index_path):
    """``torch.equal`` calls any NaN tensor unequal to itself; aligned data must pass."""
    view = NanBulkCohort(index_path).view(TOY_IDS[:4], None)
    assert torch.isnan(view[0].modalities["features"][1])


def test_an_empty_view_skips_the_spot_check(index_path):
    cohort = BulkToyCohort(index_path)
    assert len(cohort.view([], None)) == 0
    assert cohort.payload_calls == [], "nothing to compare against"


def test_the_default_is_no_bulk_path(toy):
    assert toy.payload_bulk(toy.identifiers, None) is None


# =========================================================================== #
# splitting -- identifiers, not cohorts
# =========================================================================== #


def test_split_returns_identifiers(toy):
    """Names, not row numbers -- a position means nothing outside the cohort that made it."""
    train_ids, test_ids = toy.split(test_size=0.2, random_state=0, stratify=False)

    assert all(isinstance(i, str) for i in train_ids + test_ids)
    assert len(train_ids) + len(test_ids) == len(toy)
    assert set(train_ids).isdisjoint(test_ids)


def test_split_returns_identifiers_in_cohort_order(toy):
    train_ids, _ = toy.split(test_size=0.2, random_state=0, stratify=False)
    assert train_ids == [i for i in toy.identifiers if i in set(train_ids)]


def test_asking_to_stratify_without_a_target_raises(toy):
    """A request that cannot be honoured is refused, not quietly ignored."""
    with pytest.raises(TypeError, match="needs a target to take labels from"):
        toy.split(test_size=0.2, random_state=0, stratify=True)


def test_stratify_is_required(toy):
    """No default: what a split balances on changes the numbers and leaves no trace."""
    with pytest.raises(TypeError, match="stratify"):
        toy.split(test_size=0.2, random_state=0)


def test_splitting_without_stratification_is_spelled_out(toy):
    train_ids, test_ids = toy.split(test_size=0.2, random_state=0, stratify=False)
    assert len(train_ids) + len(test_ids) == len(toy)


def test_split_asks_the_target_for_stratification_labels(index_path):
    cohort = ToyCohort(index_path, target=StubTarget())
    _, test_ids = cohort.split(test_size=0.4, random_state=0, stratify=True)
    labels = [int(i) % 2 for i in test_ids]
    assert sorted(labels) == [0, 0, 1, 1], "both classes represented in proportion"


def test_split_refuses_to_stratify_on_a_target_that_cannot(index_path):
    """Named as a capability, not assumed. The alternative is a confusing sklearn error."""

    class Bare:
        required_columns = ()

        def bind(self, identifiers, values):
            pass

        def for_(self, identifier):
            return {}

        def values_for(self, identifiers):
            return {}

    cohort = ToyCohort(index_path, target=Bare())
    with pytest.raises(TypeError, match="stratify_labels"):
        cohort.split(test_size=0.2, stratify=True)


def test_split_accepts_explicit_labels(toy):
    labels = np.array([0] * 5 + [1] * 5)
    _, test_ids = toy.split(test_size=0.4, random_state=0, stratify=labels)
    assert sorted(labels[int(i)] for i in test_ids) == [0, 0, 1, 1]


# =========================================================================== #
# views feed a DataLoader
# =========================================================================== #


def test_a_view_collates_through_a_dataloader(index_path):
    cohort = ToyCohort(index_path, target=StubTarget())
    loader = DataLoader(cohort.view(TOY_IDS[:10], None), batch_size=4, collate_fn=collate_samples)
    batch = next(iter(loader))

    assert batch.modalities["features"].shape == (4, 3)
    assert batch.target["outcome"].shape == (4,)
    assert batch.patient_id == ["0", "1", "2", "3"], "identifiers survive as a list of strings"


def test_a_view_without_a_preprocessor_reports_no_feature_names(toy):
    """Legitimate here: this cohort genuinely has nothing to fit."""
    assert toy.view(TOY_IDS[:4], None).feature_names == {}


def test_feature_names_pass_through_per_modality_untouched(index_path):
    """A composite preprocessor names several modalities; the view must not reshape.

    Wrapping them under the cohort's single name turns the dict into a list of its
    keys, so feature names silently become modality names.
    """

    class Composite:
        fitted_on = frozenset()
        feature_names = {"clinical": ["age", "sex"], "wsi": ["f0", "f1", "f2"]}

        def describe(self):
            return "composite"

    class Multi(ToyCohort):
        def fit_preprocessor(self, indices):
            return Composite()

    view = Multi(index_path, name="multimodal").view(TOY_IDS[:3], Composite())
    assert view.feature_names == {"clinical": ["age", "sex"], "wsi": ["f0", "f1", "f2"]}


def test_view_repr_says_how_much_of_the_cohort_it_covers(toy):
    assert "4 of 10 samples" in repr(toy.view(TOY_IDS[:4], None))


def test_cohort_repr_reports_size_and_name(toy):
    assert "10 samples" in repr(toy)
    assert "features" in repr(toy)


# =========================================================================== #
# errors
# =========================================================================== #


def test_the_error_types_are_runtime_errors():
    """Both are conditions a caller may reasonably catch, not programming mistakes."""
    assert issubclass(NotFittedError, RuntimeError)
    assert issubclass(LeakageError, RuntimeError)


# =========================================================================== #
# identifiers are checked, because a wrong one is silent otherwise
# =========================================================================== #


def test_an_identifier_from_another_cohort_is_refused(toy, tmp_path):
    """A split carried across cohorts is the mistake identifiers exist to expose."""
    other = tmp_path / "other.txt"
    other.write_text("a\nb\nc", encoding="utf-8")

    with pytest.raises(ValueError, match="not in this cohort"):
        toy.view(ToyCohort(other).identifiers, None)


def test_a_repeated_identifier_is_refused(toy):
    """A sample counted twice inflates n and lands on both sides of a comparison."""
    with pytest.raises(ValueError, match="more than once"):
        toy.view(["3", "5", "3"], None)


def test_check_identifiers_returns_a_plain_list(toy):
    assert toy.check_identifiers(iter(["2", "7"])) == ["2", "7"]


def test_a_view_holds_the_identifiers_it_was_given(toy):
    """No positions stored: the view's own order is the list you handed it."""
    view = toy.view(["7", "1", "4"], None)
    assert view.identifiers == ["7", "1", "4"]
    assert not hasattr(view, "indices")


def test_batch_collates_a_whole_view(toy):
    batch = toy.view(["7", "1", "4"], None).batch()
    assert batch.patient_id == ["7", "1", "4"]
    assert batch.modalities["features"].shape == (3, 3)
    assert batch.modalities["features"][:, 0].tolist() == [7.0, 1.0, 4.0]


def test_the_two_batch_routes_agree(index_path):
    """A cached cohort skips the per-sample path, so the two must not drift apart.

    The same reasoning as ``_check_cache_alignment``: wherever a bulk shortcut exists
    beside a per-sample path, the shortcut is what gets used and the check is what
    keeps it honest.
    """
    cached = BulkToyCohort(index_path, target=StubTarget()).view(["7", "1", "4"], None)
    uncached = ToyCohort(index_path, target=StubTarget()).view(["7", "1", "4"], None)
    assert cached._cache is not None and uncached._cache is None

    fast, slow = cached.batch(), uncached.batch()
    assert fast.patient_id == slow.patient_id
    assert fast.pad_mask == slow.pad_mask
    for name in slow.modalities:
        torch.testing.assert_close(fast.modalities[name], slow.modalities[name])
    for name in slow.present:
        torch.testing.assert_close(fast.present[name], slow.present[name])
    for key in slow.target:
        torch.testing.assert_close(fast.target[key], slow.target[key])


def test_a_batch_does_not_alias_the_views_cache(index_path):
    """Writing into a batch must not rewrite the features every later read returns."""
    view = BulkToyCohort(index_path).view(["7", "1", "4"], None)
    before = view[0].modalities["features"].clone()

    view.batch().modalities["features"] += 100.0

    torch.testing.assert_close(view[0].modalities["features"], before)
