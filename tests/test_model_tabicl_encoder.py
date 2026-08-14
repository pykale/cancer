"""Tests for ``kalecancer.model.embed.tabicl``.

TabICL itself is stubbed. That is deliberate rather than a shortcut: the real
checkpoint is ~100MB from Hugging Face and lives behind an optional extra, so a
test that loaded it would skip in CI and leave this module unexercised -- the
exact failure mode the tabular reader was trimmed to three formats to avoid.

What is worth testing here is not TabICL's numerics but *this wrapper's* contract:
that fitting returns a new object, that the context block is sliced back off, that
rows come out in identifier order, and that the guards fire. Those are the things
that fail silently. A stub whose representations encode the row they came from
pins all four, which the real model could not do any better.

The one property that genuinely needs the real weights -- that a context row's
representation is unaffected by what is encoded against it -- is measured in
``examples/HANCOCK_tabular/`` instead, where it can be reported as a number.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kalecancer.loaddata.base import NotFittedError
from kalecancer.model.embed.tabicl import TabICLEncoder

# --------------------------------------------------------------------------- #
# a stub standing in for TabICLClassifier
# --------------------------------------------------------------------------- #

EMBED_DIM, ROW_NUM_CLS = 8, 4
STUB_OUT_DIM = EMBED_DIM * ROW_NUM_CLS


class _StubModel:
    """Representations that carry the row's first feature, so rows are traceable.

    ``col_embedder`` passes the block through untouched and records the arguments
    it was called with; ``row_interactor`` broadcasts each row's first feature
    across the output width. Any mistake in slicing the context off, or in the
    order rows are stacked, therefore shows up as a wrong *value*, not just a
    wrong shape.
    """

    embed_dim = EMBED_DIM
    row_num_cls = ROW_NUM_CLS

    def __init__(self):
        self.col_calls: list[dict] = []
        self.training = True  # a freshly built nn.Module is in train mode

    def eval(self):
        self.training = False
        return self

    def col_embedder(self, X, *, y_train, embed_with_test, mgr_config):
        self.col_calls.append({"embed_with_test": embed_with_test, "train_size": y_train.shape[1]})
        return X

    def row_interactor(self, embeddings, *, mgr_config):
        first_feature = embeddings[..., :1]  # (B, T, 1)
        return first_feature.expand(*first_feature.shape[:-1], STUB_OUT_DIM).clone()


class _StubEnsembleGenerator:
    """Re-attaches the stored context ahead of the query rows, once per member."""

    def __init__(self, X_context, y_context, n_members):
        self.X_context = X_context
        self.y_context = y_context
        self.n_members = n_members

    def transform(self, X_query, mode):
        assert mode == "both"
        block = np.concatenate([self.X_context, X_query], axis=0)
        # Member m is offset by m so averaging across members is observable.
        Xs = np.stack([block + m for m in range(self.n_members)])
        ys = np.stack([self.y_context for _ in range(self.n_members)])
        return {"none": (Xs, ys)}


class _StubClassifier:
    """Enough of TabICLClassifier's fitted surface for the encoder to run."""

    instances: list[_StubClassifier] = []

    #: Counts real checkpoint reads. The encoder shares one loaded module between
    #: folds, so this must not climb with the number of fits.
    loads = 0

    def __init__(self, n_estimators=1, device=None, random_state=None, checkpoint_version=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.checkpoint_version = checkpoint_version
        self.device_ = "cpu"
        self.X_encoder_ = type("Enc", (), {"transform": staticmethod(lambda X: np.asarray(X, dtype=float))})()
        self.inference_config_ = type("Cfg", (), {"COL_CONFIG": None, "ROW_CONFIG": None})()
        _StubClassifier.instances.append(self)

    def _load_model(self):
        """Mirrors TabICLClassifier: fit() routes the checkpoint read through here."""
        _StubClassifier.loads += 1
        self.model_ = _StubModel()
        self.model_config_ = {"embed_dim": EMBED_DIM}
        self.model_path_ = "stub.ckpt"

    def fit(self, X, y):
        self._load_model()
        self.fit_X, self.fit_y = np.asarray(X, dtype=float), np.asarray(y)
        self.n_features_in_ = self.fit_X.shape[1]
        self.ensemble_generator_ = _StubEnsembleGenerator(self.fit_X, self.fit_y, self.n_estimators)
        return self


@pytest.fixture(autouse=True)
def stub_tabicl(monkeypatch):
    """Swap the lazily-imported TabICLClassifier for the stub, for every test here.

    Also clears the shared-checkpoint cache. It is keyed by ``(checkpoint, device)``
    and lives for the process, so without this a stub loaded by one test would be
    handed to the next -- and its recorded ``col_calls`` read as this test's.
    """
    _StubClassifier.instances.clear()
    _StubClassifier.loads = 0
    monkeypatch.setattr("kalecancer.model.embed.tabicl._LOADED", {})
    monkeypatch.setattr("kalecancer.model.embed.tabicl._require_tabicl", lambda: _StubClassifier)
    return _StubClassifier


@pytest.fixture
def folds(cohort):
    """A fitted train fold and a test fold carrying the train fold's statistics."""
    train, test = cohort.split(test_size=0.25, random_state=0)
    fitted_train = train.fit_transform()
    return fitted_train, fitted_train.transform(test)


def first_features(dataset) -> np.ndarray:
    """Each row's leading feature value, in identifier order."""
    return np.array([dataset.get_by_id(i)[0].item() for i in dataset.identifiers], dtype=float)


# --------------------------------------------------------------------------- #
# the context block must be sliced back off
# --------------------------------------------------------------------------- #


def test_encode_returns_query_rows_not_context_rows(folds):
    """Returning the context block instead would hand back the wrong patients entirely.

    The forward pass sees context rows *and* query rows concatenated. Slicing the
    wrong side, or off by the context size, still yields a plausibly-shaped matrix,
    so nothing downstream would notice.
    """
    train, test = folds
    encoder = TabICLEncoder().fit(train)

    encoded = encoder.encode(test)

    assert encoded.shape == (len(test), STUB_OUT_DIM)
    np.testing.assert_allclose(encoded[:, 0].numpy(), first_features(test), rtol=1e-6)


def test_encode_preserves_identifier_order(folds):
    """Row i of the output must be identifiers[i], or every label is attached to the wrong patient."""
    train, test = folds
    encoder = TabICLEncoder().fit(train)

    reordered = test.subset(list(reversed(range(len(test)))))
    encoded = encoder.encode(reordered)

    assert reordered.identifiers == list(reversed(test.identifiers))
    np.testing.assert_allclose(encoded[:, 0].numpy(), first_features(reordered), rtol=1e-6)


def test_encode_of_a_subset_matches_those_rows_of_the_whole(folds):
    """Encoding in batches must not change the answer, or results depend on call pattern."""
    train, test = folds
    encoder = TabICLEncoder().fit(train)

    whole = encoder.encode(test)
    indices = [0, 3, 7]
    part = encoder.encode(test.subset(indices))

    torch.testing.assert_close(part, whole[indices])


# --------------------------------------------------------------------------- #
# context rows must never attend to the rows being encoded
# --------------------------------------------------------------------------- #


def test_embed_with_test_is_never_enabled(folds):
    """With this flag on, context representations shift with the query block.

    Measured on the real checkpoint, flipping it moves a context row's
    representation by ~0.27 per dimension where it should move by exactly 0. It is
    not exposed as an argument; this pins that it is not passed as True by accident.
    """
    train, test = folds
    encoder = TabICLEncoder().fit(train)

    encoder.encode(test)

    assert encoder._clf.model_.col_calls, "col_embedder was never called"
    assert all(call["embed_with_test"] is False for call in encoder._clf.model_.col_calls)


def test_context_size_passed_to_the_model_is_the_training_fold(folds):
    """The label block sets the train/test boundary; a wrong one mis-slices the output."""
    train, test = folds
    encoder = TabICLEncoder().fit(train)

    encoder.encode(test)

    assert {call["train_size"] for call in encoder._clf.model_.col_calls} == {len(train)}


# --------------------------------------------------------------------------- #
# fitting returns a new object and never mutates
# --------------------------------------------------------------------------- #


def test_fit_returns_a_new_encoder_and_leaves_the_original_unfitted(folds):
    """Folds sharing one encoder would share a context, silently leaking across them."""
    train, _ = folds
    encoder = TabICLEncoder()

    fitted = encoder.fit(train)

    assert fitted is not encoder
    assert fitted.is_fitted
    assert not encoder.is_fitted


def test_two_folds_fitted_from_one_encoder_hold_separate_contexts(cohort):
    """The whole point of clone-per-fold: fold B must not inherit fold A's context."""
    train, test = cohort.split(test_size=0.25, random_state=0)
    fold_a = train.subset(range(0, len(train), 2)).fit_transform()
    fold_b = train.subset(range(1, len(train), 2)).fit_transform()
    encoder = TabICLEncoder()

    a, b = encoder.fit(fold_a), encoder.fit(fold_b)

    assert a._clf is not b._clf
    assert a._context_size == len(fold_a)
    assert b._context_size == len(fold_b)
    assert not np.array_equal(a._clf.fit_X, b._clf.fit_X)


def test_folds_share_one_loaded_checkpoint(cohort):
    """Cloning per fold must not clone 110MB of frozen weights along with the context.

    Five folds loading their own copy costs ~550MB of identical tensors. The weights
    are read-only here -- no fine-tuning, no gradients, kv_cache off -- so one module
    serves every fold.
    """
    train, _ = cohort.split(test_size=0.25, random_state=0)
    encoder = TabICLEncoder()

    fitted = [encoder.fit(train.subset(range(i, len(train), 5)).fit_transform()) for i in range(5)]

    assert len({id(f._clf.model_) for f in fitted}) == 1
    assert _StubClassifier.loads == 1, "the checkpoint was read more than once"


def test_sharing_weights_does_not_share_the_context(cohort):
    """The whole point of the clone survives the memory optimisation.

    Sharing the module must not let folds see each other's in-context rows -- that
    would make every fold report the last fold's context, silently.
    """
    train, _ = cohort.split(test_size=0.25, random_state=0)
    encoder = TabICLEncoder()

    fitted = [encoder.fit(train.subset(range(i, len(train), 5)).fit_transform()) for i in range(5)]

    assert len({id(f._clf) for f in fitted}) == 5
    assert not np.array_equal(fitted[0]._clf.fit_X, fitted[1]._clf.fit_X)


def test_different_checkpoints_do_not_share_a_module(folds):
    """The cache is keyed, not global; two checkpoints in one process stay distinct."""
    train, _ = folds

    a = TabICLEncoder(checkpoint="a.ckpt").fit(train)
    b = TabICLEncoder(checkpoint="b.ckpt").fit(train)

    assert a._clf.model_ is not b._clf.model_
    assert _StubClassifier.loads == 2


def test_encoding_puts_the_shared_module_in_eval_mode(folds):
    """ColEmbedding.forward branches on self.training, and the module is shared.

    Left in train mode it would take a different code path and return different
    numbers -- and one encoder could change what another returns.
    """
    train, test = folds
    encoder = TabICLEncoder().fit(train)
    encoder._clf.model_.training = True

    encoder.encode(test)

    assert encoder._clf.model_.training is False


def test_context_labels_come_from_the_target_event_indicator(folds):
    """A context conditioned on the wrong column trains fine and means nothing."""
    train, _ = folds

    fitted = TabICLEncoder(context_label="event").fit(train)

    expected = train.target.events_for(train.identifiers)
    np.testing.assert_array_equal(fitted._clf.fit_y, expected.astype(int))


# --------------------------------------------------------------------------- #
# ensembling
# --------------------------------------------------------------------------- #


def test_single_estimator_is_the_default(folds):
    """Averaging representations across permutations is unvalidated; it must be opt-in."""
    train, _ = folds

    fitted = TabICLEncoder().fit(train)

    assert fitted._clf.n_estimators == 1


def test_multiple_estimators_are_averaged(folds):
    """Members offset by 0 and 1 must average to +0.5, not concatenate or overwrite."""
    train, test = folds

    one = TabICLEncoder(n_estimators=1).fit(train).encode(test)
    two = TabICLEncoder(n_estimators=2).fit(train).encode(test)

    assert two.shape == one.shape
    torch.testing.assert_close(two, one + 0.5)


# --------------------------------------------------------------------------- #
# loud failure
# --------------------------------------------------------------------------- #


def test_encode_before_fit_raises(folds):
    """Encoding with no context cannot produce a meaningful representation."""
    _, test = folds

    with pytest.raises(NotFittedError, match="no in-context example set"):
        TabICLEncoder().encode(test)


def test_out_dim_before_fit_raises():
    """The width comes from the checkpoint, which is not loaded until fit."""
    with pytest.raises(NotFittedError, match="no in-context example set"):
        _ = TabICLEncoder().out_dim


def test_fit_without_a_target_raises(make_dataset):
    """TabICL's column embedder is target-aware; a context with no labels is not a context."""
    dataset = make_dataset(target=None)

    with pytest.raises(ValueError, match="needs a dataset with a target"):
        TabICLEncoder().fit(dataset.fit_transform())


def test_fit_with_a_target_that_cannot_supply_event_labels_raises(cohort, folds):
    """``Target`` promises only bind/for_; event language is not universal to targets.

    A target that satisfies the protocol but has no ``events_for`` must be turned
    away by name, not fail later on a missing attribute inside the context build.
    """
    train, _ = folds
    train.target = type("LabelTarget", (), {"bind": lambda *a: None, "for_": lambda *a: {}})()

    with pytest.raises(TypeError, match="needs a target providing events_for"):
        TabICLEncoder().fit(train)


def test_fit_on_a_dataset_with_unfitted_transforms_raises(cohort):
    """Otherwise the context is built from untransformed columns without complaint."""
    with pytest.raises(NotFittedError):
        TabICLEncoder().fit(cohort)


def test_encode_an_empty_dataset_raises(folds):
    """An empty query block would return a (0, d) matrix and quietly train on nothing."""
    train, test = folds
    encoder = TabICLEncoder().fit(train)

    with pytest.raises(ValueError, match="empty dataset"):
        encoder.encode(test.subset([]))


def test_encoding_a_differently_shaped_dataset_raises(cohort, make_dataset):
    """A fold-fitted one-hot encoder can emit a column the next fold's does not.

    Named for the realistic cause: ``handle_unknown='ignore'`` keeps a rare category
    from raising, but a dataset fitted *separately* rather than transformed by the
    training fold ends up a different width. Without this guard that surfaces as a
    shape error inside the transformer, nowhere near the transform that caused it.
    """
    train, _ = cohort.split(test_size=0.25, random_state=0)
    encoder = TabICLEncoder().fit(train.fit_transform())

    narrower = make_dataset(continuous=["age"], categorical=[])
    narrower = narrower.fit_transform()

    with pytest.raises(ValueError, match="was fitted on .* features but was given"):
        encoder.encode(narrower)


def test_unknown_context_label_raises():
    """A closed vocabulary of two strings, validated where the typo was written."""
    with pytest.raises(ValueError, match="Unknown context_label"):
        TabICLEncoder(context_label="events")


def test_time_context_label_raises_rather_than_silently_using_events():
    """Falling back to the event indicator would answer a different question than asked."""
    with pytest.raises(NotImplementedError, match="TabICL regressor"):
        TabICLEncoder(context_label="time")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def test_out_dim_is_read_from_the_checkpoint(folds):
    train, _ = folds

    assert TabICLEncoder().fit(train).out_dim == STUB_OUT_DIM


def test_repr_shows_fitted_state_and_context_size(folds):
    train, _ = folds

    assert repr(TabICLEncoder()) == "TabICLEncoder(context_label='event' | unfitted)"
    assert repr(TabICLEncoder().fit(train)) == (
        f"TabICLEncoder(context_label='event' | context {len(train)} rows -> {STUB_OUT_DIM}d)"
    )


def test_checkpoint_is_forwarded_only_when_given(folds):
    """Passing None explicitly would override TabICL's own default."""
    train, _ = folds

    assert TabICLEncoder().fit(train)._clf.checkpoint_version is None
    assert TabICLEncoder(checkpoint="x.ckpt").fit(train)._clf.checkpoint_version == "x.ckpt"
