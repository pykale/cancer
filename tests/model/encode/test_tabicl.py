"""TabICLEncoder tests.

Most tests run against a tiny, randomly initialised TabICL saved as a checkpoint, so they need the ``tabicl`` package
but no download. Only the comparison with upstream ``TabICLClassifier`` needs the pretrained checkpoint, and it is
skipped unless that checkpoint is already in the Hugging Face cache.
"""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from torch import nn

pytest.importorskip("tabicl")

from tabicl import TabICLClassifier  # noqa: E402
from tabicl._model.tabicl import TabICL  # noqa: E402
from tabicl._sklearn.preprocessing import PreprocessingPipeline  # noqa: E402

import kalecancer.model.encode.tabicl as tabicl_module  # noqa: E402
from kalecancer.loaddata import Classification, TimeToEvent  # noqa: E402
from kalecancer.model import TabICLEncoder  # noqa: E402

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
pretrained = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache("jingang/TabICL", CHECKPOINT), str),
    reason=f"{CHECKPOINT} is not in the Hugging Face cache",
)

TINY = dict(
    max_classes=10,
    embed_dim=8,
    col_num_blocks=1,
    col_nhead=2,
    col_num_inds=4,
    row_num_blocks=1,
    row_nhead=2,
    row_num_cls=2,
    icl_num_blocks=1,
    icl_nhead=2,
    dropout=0.0,
)

N_CONTEXT = 120


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_CONTEXT + 16, 5)).astype(np.float32)
    x[:, 1] = x[:, 1] * 10 + 50
    labels = (x[:, 0] + 0.5 * rng.normal(size=len(x)) > 0).astype(np.int64)
    ids = [f"p{k:03d}" for k in range(len(x))]
    return x, labels, ids


def context(x, labels, ids, key):
    """The rows, a target whose strata are ``labels`` (survival events or class labels), and the ids."""
    index = pd.Index(list(ids))
    if key == "event":
        target = TimeToEvent(time=pd.Series(1.0, index=index), event=pd.Series(labels.astype(bool), index=index))
    else:
        target = Classification(pd.Series(labels, index=index), classes=sorted(set(labels.tolist())))
    return torch.tensor(x), target, list(ids)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = TabICL(**TINY)
    # TabICL zero-initialises its attention output projections, so a newly built model ignores its context and a
    # leakage test would pass whatever the encoder does.
    for parameter in model.parameters():
        nn.init.normal_(parameter, std=0.3)
    path = tmp_path_factory.mktemp("tabicl") / "tiny.ckpt"
    torch.save({"config": TINY, "state_dict": model.state_dict()}, path)
    return str(path)


@pytest.fixture
def encoder(tiny_checkpoint, monkeypatch):
    """Build a ``TabICLEncoder`` on the tiny checkpoint."""
    monkeypatch.setattr(tabicl_module, "hf_hub_download", lambda *args, **kwargs: tiny_checkpoint)

    def build(trainable=(), output="row"):
        return TabICLEncoder(
            checkpoint="tiny.ckpt", output=output, trainable=list(trainable), context_folds=5, random_state=0
        )

    return build


def fitted(build, table, key="event", **kwargs):
    x, labels, ids = table
    model = build(**kwargs)
    model.fit(*context(x[:N_CONTEXT], labels[:N_CONTEXT], ids[:N_CONTEXT], key=key))
    return model


@pytest.fixture
def frozen(encoder, table):
    return fitted(encoder, table)


def upstream_row_embeddings(context_x, context_labels, queries):
    """The path the encoder re-implements: TabICLClassifier's input encoding and "none" ensemble member, then the
    column and row stages. With one estimator there is no feature or class shuffling."""
    clf = TabICLClassifier(n_estimators=1, random_state=0, device="cpu").fit(context_x, context_labels)
    members = clf.ensemble_generator_.transform(clf.X_encoder_.transform(queries), mode="both")
    assert list(members) == ["none"]
    ((block, labels),) = members.values()
    with torch.no_grad():
        rows = torch.as_tensor(block, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.float32)
        config = clf.inference_config_
        columns = clf.model_.col_embedder(rows, y_train=y, embed_with_test=False, mgr_config=config.COL_CONFIG)
        return clf.model_.row_interactor(columns, mgr_config=config.ROW_CONFIG)[0, y.shape[1] :]


@pretrained
def test_unseen_rows_match_upstream_row_embeddings(table):
    x, labels, ids = table
    model = TabICLEncoder(checkpoint=CHECKPOINT, output="row", trainable=[], context_folds=5, random_state=0)
    model.fit(*context(x[:N_CONTEXT], labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
    upstream = upstream_row_embeddings(x[:N_CONTEXT], labels[:N_CONTEXT], x[N_CONTEXT:])
    with torch.no_grad():
        ours = model(torch.tensor(x[N_CONTEXT:]), ids[N_CONTEXT:])
    assert ours.shape == (16, 512)
    # Upstream's inference kernels and the training branches differ by float32 noise (~2e-5). Semantic mistakes are
    # far larger: dropping a single context row moves the embeddings well beyond this tolerance.
    torch.testing.assert_close(ours, upstream, atol=1e-4, rtol=0)
    one_row_less = upstream_row_embeddings(x[1:N_CONTEXT], labels[1:N_CONTEXT], x[N_CONTEXT:])
    assert (one_row_less - upstream).abs().max() > 1e-3


def test_normaliser_matches_upstream_transform(table, frozen):
    x, _, _ = table
    upstream = PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit(
        x[:N_CONTEXT].astype(np.float64)
    )
    queries = x[N_CONTEXT:] * 3
    np.testing.assert_allclose(frozen._normalise(torch.tensor(queries)).numpy(), upstream.transform(queries), atol=1e-6)


@pytest.mark.parametrize("output", ["row", "icl"])
def test_nothing_in_a_training_rows_own_fold_reaches_its_embedding(encoder, table, output):
    x, _, ids = table
    model = fitted(encoder, table, output=output)
    fold = model.context_fold
    other = int((fold != fold[0]).nonzero()[0])  # a training patient in another fold
    queries = torch.tensor(np.concatenate([x[:1], x[other : other + 1], x[N_CONTEXT : N_CONTEXT + 1]]))
    query_ids = [ids[0], ids[other], "someone-else"]

    # Change everything about patient 0's fold that the model reads: the labels and the normalised rows.
    changed = copy.deepcopy(model)
    own = fold == fold[0]
    changed.context_y[own] = 1 - changed.context_y[own]
    changed.context_z[own] += 1.0
    with torch.no_grad():
        before, after = model(queries, query_ids), changed(queries, query_ids)

    torch.testing.assert_close(after[0], before[0], atol=1e-6, rtol=0)
    # The same change reaches a patient in another fold and an unseen patient, so the context is in use.
    assert not torch.allclose(after[1], before[1], atol=1e-4)
    assert not torch.allclose(after[2], before[2], atol=1e-4)


def test_a_context_id_with_a_different_row_raises(table, frozen):
    x, _, ids = table
    with pytest.raises(ValueError, match="rows differ from the context rows"):
        frozen(torch.tensor(x[:1]) + 1, ids[:1])


def test_gradients_reach_exactly_the_trainable_stages(encoder, table):
    x, _, ids = table
    model = fitted(encoder, table, trainable=["col", "row"])
    queries = torch.tensor(x[N_CONTEXT - 8 :], requires_grad=True)
    model(queries, ids[N_CONTEXT - 8 :]).sum().backward()
    assert all(p.grad is not None for p in model.tabicl.col_embedder.parameters())
    assert all(p.grad is not None for p in model.tabicl.row_interactor.parameters())
    assert all(p.grad is None for p in model.tabicl.icl_predictor.parameters())
    assert queries.grad is not None and torch.isfinite(queries.grad).all()


def test_frozen_encoder_has_no_trainable_parameters(frozen):
    assert not any(p.requires_grad for p in frozen.parameters())


def test_batching_does_not_change_embeddings(table, frozen):
    x, _, ids = table
    with torch.no_grad():
        batch = frozen(torch.tensor(x[:20]), ids[:20])
        one_by_one = torch.cat([frozen(torch.tensor(x[k : k + 1]), ids[k : k + 1]) for k in range(20)])
        repeated = frozen(torch.tensor(np.concatenate([x[:2], x[:2]])), ids[:2] + ids[:2])
        unseen = torch.tensor(x[N_CONTEXT : N_CONTEXT + 4])
        alone = frozen(unseen, ids[N_CONTEXT : N_CONTEXT + 4])
        mixed = frozen(torch.cat([unseen, torch.tensor(x[:4])]), ids[N_CONTEXT : N_CONTEXT + 4] + ids[:4])[:4]
    torch.testing.assert_close(batch, one_by_one, atol=1e-5, rtol=0)
    torch.testing.assert_close(repeated[:2], repeated[2:])
    torch.testing.assert_close(alone, mixed, atol=1e-5, rtol=0)


def test_deepcopy_and_state_dict_round_trip_weights_and_context(encoder, table):
    x, _, ids = table
    model = fitted(encoder, table, trainable=["row"])
    with torch.no_grad():
        next(model.tabicl.row_interactor.parameters()).add_(0.05)
    queries, query_ids = torch.tensor(x[N_CONTEXT - 4 :]), ids[N_CONTEXT - 4 :]
    with torch.no_grad():
        expected = model(queries, query_ids)
        copied = copy.deepcopy(model)
        torch.testing.assert_close(copied(queries, query_ids), expected)
        fresh = encoder(trainable=["row"])
        assert not fresh.context_ids
        fresh.load_state_dict(model.state_dict())
        assert fresh.context_ids == model.context_ids
        torch.testing.assert_close(fresh(queries, query_ids), expected)


def test_outer_bf16_autocast_does_not_change_outputs(table, frozen):
    x, _, ids = table
    queries = torch.tensor(x[N_CONTEXT - 4 :])
    with torch.no_grad():
        reference = frozen(queries, ids[N_CONTEXT - 4 :])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed = frozen(queries, ids[N_CONTEXT - 4 :])
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(mixed, reference, atol=1e-6, rtol=0)


def test_icl_output_has_the_same_width(encoder, table, frozen):
    x, _, ids = table
    model = fitted(encoder, table, key="label", output="icl")
    with torch.no_grad():
        icl = model(torch.tensor(x[N_CONTEXT:]), ids[N_CONTEXT:])
        row = frozen(torch.tensor(x[N_CONTEXT:]), ids[N_CONTEXT:])
    assert icl.shape == row.shape == (16, TINY["embed_dim"] * TINY["row_num_cls"])


def test_guards(encoder, table):
    x, labels, ids = table
    with pytest.raises(ValueError, match="'icl' cannot be trainable"):
        encoder(trainable=["icl"], output="row")
    with pytest.raises(RuntimeError, match="fit must be called before forward"):
        encoder()(torch.tensor(x[:1]), ids[:1])
    # 24 classes of 5 rows each: enough rows per fold, but more classes than the checkpoint handles
    with pytest.raises(ValueError, match=r"the context has 24 classes, but .* handles at most 10"):
        encoder().fit(*context(x[:N_CONTEXT], np.arange(N_CONTEXT) % 24, ids[:N_CONTEXT], key="label"))
    rare = (np.arange(N_CONTEXT) < 3).astype(np.int64)
    with pytest.raises(ValueError, match=r"fewer rows than context_folds=5"):
        encoder().fit(*context(x[:N_CONTEXT], rare, ids[:N_CONTEXT], key="label"))
    model = encoder()
    with_nan = x[:N_CONTEXT].copy()
    with_nan[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        model.fit(*context(with_nan, labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
    constant = x[:N_CONTEXT].copy()
    constant[:, 4] = 1.0
    with pytest.raises(ValueError, match=r"columns \[4\] are constant"):
        model.fit(*context(constant, labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
