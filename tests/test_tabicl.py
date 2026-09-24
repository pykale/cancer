import copy

import numpy as np
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"

if not isinstance(try_to_load_from_cache("jingang/TabICL", CHECKPOINT), str):
    pytest.skip(f"{CHECKPOINT} is not in the Hugging Face cache", allow_module_level=True)

from tabicl import TabICLClassifier  # noqa: E402
from tabicl._sklearn.preprocessing import PreprocessingPipeline  # noqa: E402

from kalecancer.model import TabICLEncoder  # noqa: E402

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
    target = {"event": torch.tensor(labels.astype(bool))} if key == "event" else {"label": torch.tensor(labels)}
    return torch.tensor(x), target, list(ids)


def encoder(trainable=(), output="row", context_label="event"):
    return TabICLEncoder(
        checkpoint=CHECKPOINT,
        output=output,
        trainable=list(trainable),
        context_label=context_label,
        context_folds=5,
        random_state=0,
    )


@pytest.fixture(scope="module")
def frozen(table):
    x, labels, ids = table
    model = encoder()
    model.fit(*context(x[:N_CONTEXT], labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
    return model


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


def test_unseen_rows_match_upstream_row_embeddings(table, frozen):
    x, labels, ids = table
    upstream = upstream_row_embeddings(x[:N_CONTEXT], labels[:N_CONTEXT], x[N_CONTEXT:])
    with torch.no_grad():
        ours = frozen(torch.tensor(x[N_CONTEXT:]), ids[N_CONTEXT:])
    assert ours.shape == (16, frozen.out_dim) == (16, 512)
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


def test_gradients_reach_exactly_the_trainable_stages(table):
    x, labels, ids = table
    model = encoder(trainable=["col", "row"])
    model.fit(*context(x[:N_CONTEXT], labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
    queries = torch.tensor(x[N_CONTEXT - 8 :], requires_grad=True)
    model(queries, ids[N_CONTEXT - 8 :]).sum().backward()
    assert all(p.grad is not None for p in model.tabicl.col_embedder.parameters())
    assert all(p.grad is not None for p in model.tabicl.row_interactor.parameters())
    assert all(p.grad is None for p in model.tabicl.icl_predictor.parameters())
    assert queries.grad is not None and torch.isfinite(queries.grad).all()


def test_frozen_encoder_has_no_trainable_parameters(frozen):
    assert not any(p.requires_grad for p in frozen.parameters())


def test_training_rows_see_a_context_without_their_fold(table, frozen):
    x, _, ids = table
    with torch.no_grad():
        as_training_row = frozen(torch.tensor(x[:1]), ids[:1])
        as_unseen_row = frozen(torch.tensor(x[:1]), ["someone-else"])
        unseen = torch.tensor(x[N_CONTEXT : N_CONTEXT + 4])
        alone = frozen(unseen, ids[N_CONTEXT : N_CONTEXT + 4])
        mixed = frozen(torch.cat([unseen, torch.tensor(x[:4])]), ids[N_CONTEXT : N_CONTEXT + 4] + ids[:4])[:4]
    assert not torch.allclose(as_training_row, as_unseen_row, atol=1e-4)
    torch.testing.assert_close(alone, mixed, atol=1e-5, rtol=0)


def test_batching_does_not_change_embeddings(table, frozen):
    x, _, ids = table
    with torch.no_grad():
        batch = frozen(torch.tensor(x[:20]), ids[:20])
        one_by_one = torch.cat([frozen(torch.tensor(x[k : k + 1]), ids[k : k + 1]) for k in range(20)])
        repeated = frozen(torch.tensor(np.concatenate([x[:2], x[:2]])), ids[:2] + ids[:2])
    torch.testing.assert_close(batch, one_by_one, atol=1e-5, rtol=0)
    torch.testing.assert_close(repeated[:2], repeated[2:])


def test_deepcopy_and_state_dict_round_trip_keep_weights_and_context(table):
    x, labels, ids = table
    model = encoder(trainable=["row"])
    model.fit(*context(x[:N_CONTEXT], labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
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


def test_icl_output_has_the_same_width(table):
    x, labels, ids = table
    model = encoder(output="icl", context_label="label")
    model.fit(*context(x[:N_CONTEXT], labels[:N_CONTEXT], ids[:N_CONTEXT], key="label"))
    with torch.no_grad():
        assert model(torch.tensor(x[N_CONTEXT:]), ids[N_CONTEXT:]).shape == (16, 512)


def test_guards(table):
    x, labels, ids = table
    with pytest.raises(ValueError, match="'icl' cannot be trainable"):
        encoder(trainable=["icl"], output="row")
    with pytest.raises(ValueError, match=r"context labels must be class ids in \[0, "):
        encoder(context_label="label").fit(
            *context(x[:N_CONTEXT], labels[:N_CONTEXT] + 100, ids[:N_CONTEXT], key="label")
        )
    model = encoder()
    with_nan = x[:N_CONTEXT].copy()
    with_nan[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        model.fit(*context(with_nan, labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
    constant = x[:N_CONTEXT].copy()
    constant[:, 4] = 1.0
    with pytest.raises(ValueError, match=r"columns \[4\] are constant"):
        model.fit(*context(constant, labels[:N_CONTEXT], ids[:N_CONTEXT], key="event"))
