"""Tests for ``kalecancer.model.embed.tabicl``.

TabICL itself is stubbed. That is deliberate rather than a shortcut: the real
checkpoint is ~100MB from Hugging Face and lives behind an optional extra, so a
test that loaded it would skip in CI and leave this module unexercised.

What is worth testing here is not TabICL's numerics but *this wrapper's* contract:
that the context block is sliced back off, that rows come out in the order they went
in, that frozen and trainable differ only in ``requires_grad`` and weight sharing,
and that the guards fire. A stub whose representations encode the row they came from
pins the ordering ones, which the real model could not do any better.

The properties that genuinely need the real weights -- how much a context change
moves a representation, and that the rows being embedded do not see each other --
are measured in ``examples/HANCOCK_tabular/`` and recorded in the module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from kalecancer.model.embed.tabicl import TabICLEmbedder

# --------------------------------------------------------------------------- #
# a stub standing in for TabICLClassifier
# --------------------------------------------------------------------------- #

EMBED_DIM, ROW_NUM_CLS = 8, 4
STUB_OUT_DIM = EMBED_DIM * ROW_NUM_CLS


class _StubModel(nn.Module):
    """Representations that carry the row's first feature, so rows are traceable.

    ``col_embedder`` passes the block through untouched and records the arguments it
    was called with; ``row_interactor`` broadcasts each row's first feature across the
    output width, scaled by a real parameter so gradients have somewhere to land. Any
    mistake in slicing the context off, or in the order rows are stacked, therefore
    shows up as a wrong *value*, not just a wrong shape.
    """

    embed_dim = EMBED_DIM
    row_num_cls = ROW_NUM_CLS

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.col_calls: list[dict] = []

    def col_embedder(self, X, *, y_train, embed_with_test, mgr_config):
        self.col_calls.append({"embed_with_test": embed_with_test, "context_size": y_train.shape[1]})
        return X

    def row_interactor(self, embeddings, *, mgr_config):
        first_feature = embeddings[..., :1] * self.scale  # (B, T, 1)
        return first_feature.expand(*first_feature.shape[:-1], STUB_OUT_DIM)


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
    """Enough of TabICLClassifier's fitted surface for the embedder to run."""

    instances: list[_StubClassifier] = []

    def __init__(self, n_estimators=1, device=None, random_state=None, checkpoint_version=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.checkpoint_version = checkpoint_version
        self.device_ = "cpu"
        self.X_encoder_ = type("Enc", (), {"transform": staticmethod(lambda X: np.asarray(X, dtype=float))})()
        self.inference_config_ = type("Cfg", (), {"COL_CONFIG": None, "ROW_CONFIG": None})()
        _StubClassifier.instances.append(self)

    def fit(self, X, y):
        """Mirrors TabICLClassifier: loading the checkpoint happens here, not in __init__."""
        self.model_ = _StubModel()
        self.fit_X, self.fit_y = np.asarray(X, dtype=float), np.asarray(y)
        self.ensemble_generator_ = _StubEnsembleGenerator(self.fit_X, self.fit_y, self.n_estimators)
        return self


@pytest.fixture(autouse=True)
def stub_tabicl(monkeypatch):
    """Swap the lazily-imported TabICLClassifier for the stub, for every test here.

    Also clears the shared-backbone cache. It is keyed by ``(checkpoint, device)`` and
    lives for the process, so without this a stub shared by one test would be handed
    to the next -- and its recorded ``col_calls`` read as this test's.
    """
    _StubClassifier.instances.clear()
    monkeypatch.setattr("kalecancer.model.embed.tabicl._BACKBONES", {})
    monkeypatch.setattr("kalecancer.model.embed.tabicl._require_tabicl", lambda: _StubClassifier)
    return _StubClassifier


@pytest.fixture
def context():
    """A 12-row context whose first feature is the row number, so rows are traceable."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    X[:, 0] = np.arange(12, dtype=float)
    return X, np.array([0, 1] * 6)


@pytest.fixture
def query():
    """Query rows whose first feature is 100, 101, ... -- distinct from any context row."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 3))
    X[:, 0] = 100 + np.arange(5, dtype=float)
    return torch.tensor(X, dtype=torch.float32)


def frozen(context, **kwargs):
    X, y = context
    return TabICLEmbedder(X, y, trainable=False, random_state=0, **kwargs)


# --------------------------------------------------------------------------- #
# the context block must be sliced back off, in order
# --------------------------------------------------------------------------- #


def test_forward_returns_query_rows_not_context_rows(context, query):
    """The context is embedded alongside the query; only the query comes back."""
    out = frozen(context)(query)

    assert out.shape == (len(query), STUB_OUT_DIM)
    # The stub encodes each row's first feature, so 100..104 proves these are the
    # query rows and not the leading rows of the context block.
    assert out[:, 0].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_forward_preserves_row_order(context, query):
    """Row i out corresponds to row i in, or every prediction is against the wrong patient."""
    shuffled = query[[3, 0, 4, 1, 2]]
    out = frozen(context)(shuffled)
    assert out[:, 0].tolist() == [103.0, 100.0, 104.0, 101.0, 102.0]


def test_a_subset_matches_those_rows_of_the_whole(context, query):
    """Embedding in parts must not change the answer, or results depend on call pattern."""
    embedder = frozen(context)
    whole = embedder(query)
    part = embedder(query[[0, 3]])
    torch.testing.assert_close(part, whole[[0, 3]])


def test_needs_full_batch_is_declared(context):
    """A training loop reads this to decide it must not minibatch this embedder."""
    assert TabICLEmbedder.needs_full_batch is True
    assert frozen(context).needs_full_batch is True


# --------------------------------------------------------------------------- #
# context rows must never attend to the rows being embedded
# --------------------------------------------------------------------------- #


def test_embed_with_test_is_never_enabled(context, query):
    """With this flag on, context representations shift with the query block.

    Measured on the real checkpoint, flipping it moves a context row's representation
    by ~0.27 per dimension where it should move by exactly 0.
    """
    embedder = frozen(context)
    embedder(query)
    assert [call["embed_with_test"] for call in embedder.backbone.col_calls] == [False]


def test_the_context_handed_in_is_the_context_used(context, query):
    """The model must see exactly the rows given, not the query block or the union."""
    X, _ = context
    embedder = frozen(context)
    embedder(query)
    assert [call["context_size"] for call in embedder.backbone.col_calls] == [len(X)]


# --------------------------------------------------------------------------- #
# the context is data, and it is per-fold
# --------------------------------------------------------------------------- #


def test_two_embedders_hold_separate_contexts(context, query):
    """One fold's context must not reach another's, however the weights are shared."""
    X, y = context
    big = TabICLEmbedder(X, y, trainable=False, random_state=0)
    small = TabICLEmbedder(X[:6], y[:6], trainable=False, random_state=0)

    assert (big.context_size, small.context_size) == (12, 6)

    # One shared backbone, so one shared record: the sequence is what distinguishes
    # them, and it must show each embedder using its own context.
    big(query)
    small(query)
    assert [call["context_size"] for call in big.backbone.col_calls] == [12, 6]


def test_context_labels_are_whatever_was_passed(context):
    """The embedder conditions on the array it was given and asks nothing about it.

    It never reaches for a target, so a cohort with no survival target -- or none at
    all -- can still be embedded.
    """
    X, _ = context
    labels = np.array([7, 9] * 6)
    TabICLEmbedder(X, labels, trainable=False, random_state=0)
    assert np.array_equal(_StubClassifier.instances[-1].fit_y, labels)


def test_labels_may_come_from_a_survival_target(cohort):
    """The survival path still works -- it is now the caller's wiring, not a coupling."""
    train_ids, _ = cohort.split(test_size=0.25, random_state=0, stratify=True)
    train = cohort.view(train_ids, cohort.fit_preprocessor(train_ids))
    context = train.batch()

    embedder = TabICLEmbedder(
        context_x=context.modalities["clinical"],
        context_y=context.target["event"],
        trainable=False,
        random_state=0,
    )
    assert embedder.context_size == len(train)
    assert embedder(context.modalities["clinical"]).shape == (len(train), STUB_OUT_DIM)


# --------------------------------------------------------------------------- #
# frozen and trainable differ only in requires_grad and weight sharing
# --------------------------------------------------------------------------- #


def test_frozen_embedders_share_one_backbone(context):
    """110MB of read-only weights, and cross-validation builds one embedder per fold."""
    X, y = context
    first = TabICLEmbedder(X, y, trainable=False, random_state=0)
    second = TabICLEmbedder(X[:6], y[:6], trainable=False, random_state=0)
    assert first.backbone is second.backbone


def test_trainable_embedders_never_share_a_backbone(context):
    """Sharing weights being fine-tuned would have each fold train the others'."""
    X, y = context
    first = TabICLEmbedder(X, y, trainable=True, random_state=0)
    second = TabICLEmbedder(X, y, trainable=True, random_state=0)
    frozen_one = TabICLEmbedder(X, y, trainable=False, random_state=0)

    assert first.backbone is not second.backbone
    assert first.backbone is not frozen_one.backbone


def test_different_checkpoints_do_not_share_a_backbone(context):
    X, y = context
    first = TabICLEmbedder(X, y, trainable=False, random_state=0, checkpoint="v1")
    second = TabICLEmbedder(X, y, trainable=False, random_state=0, checkpoint="v2")
    assert first.backbone is not second.backbone


def test_frozen_weights_take_no_gradient(context, query):
    embedder = frozen(context)
    assert not any(p.requires_grad for p in embedder.parameters())
    assert not embedder(query).requires_grad


def test_trainable_weights_take_gradient(context, query):
    """Gradients must reach the backbone, or `trainable=True` is a lie."""
    X, y = context
    embedder = TabICLEmbedder(X, y, trainable=True, random_state=0)
    head = nn.Linear(embedder.out_dim, 1)

    head(embedder(query)).sum().backward()
    assert embedder.backbone.scale.grad is not None
    assert embedder.backbone.scale.grad.abs().sum() > 0


def test_a_frozen_backbone_stays_in_eval_when_the_model_trains(context):
    """The real ColEmbedding branches on self.training, and the module is shared.

    Lightning calls .train() on the whole graph at each epoch, so without the override
    one fold entering training would change what every other fold computes.
    """
    embedder = frozen(context)
    embedder.train()
    assert embedder.backbone.training is False


def test_a_trainable_backbone_follows_the_surrounding_model(context):
    X, y = context
    embedder = TabICLEmbedder(X, y, trainable=True, random_state=0)
    assert embedder.backbone.training is True  # nn.Module semantics: fresh means training

    embedder.eval()
    assert embedder.backbone.training is False
    embedder.train()
    assert embedder.backbone.training is True


# --------------------------------------------------------------------------- #
# ensembling
# --------------------------------------------------------------------------- #


def test_single_estimator_is_the_default(context):
    """Averaging representations is unvalidated, so it must be asked for."""
    frozen(context)
    assert _StubClassifier.instances[-1].n_estimators == 1


def test_multiple_estimators_are_averaged(context, query):
    X, y = context
    embedder = TabICLEmbedder(X, y, trainable=False, random_state=0, n_estimators=3)
    out = embedder(query)
    # The stub offsets member m by m, so the mean over three members shifts by +1.
    assert out[:, 0].tolist() == [101.0, 102.0, 103.0, 104.0, 105.0]


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def test_out_dim_is_known_at_construction(context):
    """The projection layer that consumes it is built in the same breath."""
    assert frozen(context).out_dim == STUB_OUT_DIM


def test_a_differently_shaped_input_raises(context):
    """A one-hot encoder fitted on one fold can emit a column another's does not."""
    embedder = frozen(context)
    with pytest.raises(ValueError, match="context of 3 features but was given 5"):
        embedder(torch.zeros(4, 5))


def test_a_non_matrix_input_raises(context):
    with pytest.raises(ValueError, match=r"expects \(n_samples, n_features\)"):
        frozen(context)(torch.zeros(4))


def test_an_empty_context_raises(context):
    _, y = context
    with pytest.raises(ValueError, match="context_x is empty"):
        TabICLEmbedder(np.zeros((0, 3)), y[:0], trainable=False, random_state=0)


def test_a_one_dimensional_context_raises(context):
    _, y = context
    with pytest.raises(ValueError, match="must be 2-D"):
        TabICLEmbedder(np.zeros(12), y, trainable=False, random_state=0)


def test_labels_that_do_not_line_up_with_the_context_raise(context):
    X, y = context
    with pytest.raises(ValueError, match="6 labels for 12 context rows"):
        TabICLEmbedder(X, y[:6], trainable=False, random_state=0)


def test_continuous_labels_raise_rather_than_becoming_hundreds_of_classes(context):
    """Conditioning on survival *time* is the mistake this catches."""
    X, _ = context
    times = np.linspace(0.5, 900.25, 12)
    with pytest.raises(ValueError, match="continuous values"):
        TabICLEmbedder(X, times, trainable=False, random_state=0)


def test_trainable_and_random_state_must_be_stated(context):
    """Both change the result and neither leaves a trace in the output."""
    X, y = context
    with pytest.raises(TypeError, match="trainable"):
        TabICLEmbedder(X, y, random_state=0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="random_state"):
        TabICLEmbedder(X, y, trainable=False)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


def test_checkpoint_is_forwarded_only_when_given(context):
    """None must mean TabICL's own default, not a version string we invented."""
    frozen(context)
    assert _StubClassifier.instances[-1].checkpoint_version is None

    frozen(context, checkpoint="tabicl-v9")
    assert _StubClassifier.instances[-1].checkpoint_version == "tabicl-v9"


def test_repr_shows_the_context_and_whether_it_is_frozen(context):
    X, y = context
    assert repr(frozen(context)) == "TabICLEmbedder(context 12x3 -> 32d | frozen)"
    trainable = TabICLEmbedder(X, y, trainable=True, random_state=0)
    assert "trainable" in repr(trainable)


def test_it_satisfies_the_embedder_protocol(context):
    """Interchangeability is the point of the protocol, so it is asserted, not assumed."""
    embedder = frozen(context)
    for member in ("out_dim", "needs_full_batch", "forward"):
        assert hasattr(embedder, member), f"Embedder requires {member}"
    assert isinstance(embedder.out_dim, int)
    assert isinstance(embedder.needs_full_batch, bool)
