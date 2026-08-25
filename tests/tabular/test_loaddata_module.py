"""``CohortDataModule``: one fold's views, wrapped for Lightning.

The data module is where a fold is *assembled*, so it is where the assembly can be
checked. Two guards live here and both exist because the mistake they catch is
otherwise invisible: overlapping splits, and a preprocessor carrying held-out rows.
"""

from __future__ import annotations

import pytest
import torch

from kalecancer.loaddata.base import LeakageError
from kalecancer.loaddata.module import CohortDataModule
from kalecancer.loaddata.sample import PatientBatch
from tests.tabular.conftest import CONTINUOUS


@pytest.fixture
def folds(cohort):
    """A three-way split: inner train, inner validation, outer test."""
    train_ids, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=True)
    return train_ids[:45], train_ids[45:], test_ids


# =========================================================================== #
# construction
# =========================================================================== #


def test_batch_size_is_required(cohort, folds):
    """No default: with a Cox head it selects the risk-set approximation."""
    with pytest.raises(TypeError, match="batch_size"):
        CohortDataModule(cohort, folds[0])


@pytest.mark.parametrize("bad", [0, -1, 2.5, None, True])
def test_batch_size_must_be_a_positive_integer_or_full(cohort, folds, bad):
    with pytest.raises(ValueError, match="positive integer or 'full'"):
        CohortDataModule(cohort, folds[0], batch_size=bad)


def test_full_resolves_to_the_training_split_size(cohort, folds):
    train_ids, _, _ = folds
    dm = CohortDataModule(cohort, train_ids, batch_size="full")
    dm.setup()
    assert next(iter(dm.train_dataloader())).modalities["clinical"].shape[0] == len(train_ids)


@pytest.mark.parametrize(
    ("overlapping", "expected"),
    [("val_ids", "train and validation"), ("test_ids", "train and test")],
)
def test_overlapping_splits_are_refused(cohort, overlapping, expected):
    """Independent of any preprocessor: evaluating on trained rows is wrong regardless.

    The leakage guard cannot see this when a cohort declares no stateful
    transforms, because a passthrough claims no rows -- so this check stands on
    its own.
    """
    ten = cohort.identifiers[:10]
    with pytest.raises(ValueError, match=expected):
        CohortDataModule(cohort, ten, batch_size=4, **{overlapping: ten[:5]})


def test_batch_size_is_recorded_in_the_hyperparameters(cohort, folds):
    """Per-batch Cox averaging means a metric is not reproducible without it."""
    dm = CohortDataModule(cohort, folds[0], batch_size=16, num_workers=0)
    assert dm.hparams["batch_size"] == 16
    assert set(dm.hparams) == {
        "batch_size",
        "num_workers",
        "pin_memory",
        "shuffle",
        "drop_last",
    }


# =========================================================================== #
# setup -- fold-local fitting
# =========================================================================== #


def test_setup_fits_on_the_training_rows_only(cohort, folds):
    train_ids, val_ids, test_ids = folds
    dm = CohortDataModule(cohort, train_ids, val_ids, test_ids, batch_size=8)
    dm.setup()

    assert dm.preprocessor.fitted_on == set(train_ids)


def test_setup_builds_every_view_from_the_same_preprocessor(cohort, folds):
    dm = CohortDataModule(cohort, *folds, batch_size=8)
    dm.setup()

    assert dm.train_ds.preprocessor is dm.preprocessor
    assert dm.val_ds.preprocessor is dm.preprocessor
    assert dm.test_ds.preprocessor is dm.preprocessor


def test_setup_is_idempotent(cohort, folds):
    """Lightning may call it more than once per Trainer; refitting would be wrong."""
    dm = CohortDataModule(cohort, *folds, batch_size=8)
    dm.setup()
    first = dm.preprocessor
    dm.setup("fit")
    assert dm.preprocessor is first


def test_setup_does_not_mutate_the_cohort(cohort, folds):
    """Every fold's data module shares one cohort, so none may write to it."""
    before = cohort.frame.copy()
    for _ in range(3):
        CohortDataModule(cohort, *folds, batch_size=8).setup()
    assert cohort.frame.equals(before)


def test_held_out_rows_are_standardised_with_training_statistics(cohort, folds):
    """End to end through the data module, not just through the cohort."""
    train_ids, _, test_ids = folds
    dm = CohortDataModule(cohort, train_ids, test_ids=test_ids, batch_size="full")
    dm.setup()

    train_batch = next(iter(dm.train_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))
    n = len(CONTINUOUS)

    assert train_batch.modalities["clinical"][:, :n].mean().item() == pytest.approx(0.0, abs=1e-5)
    assert abs(test_batch.modalities["clinical"][:, :n].mean().item()) > 1e-4


# =========================================================================== #
# the leakage guard
# =========================================================================== #


def test_a_supplied_preprocessor_fitted_on_held_out_rows_is_refused(cohort, folds):
    """The one path by which held-out statistics can reach a fold.

    In the default flow this cannot happen -- setup fits on ``train_ids`` and the
    splits are already known to be disjoint. Supplying a preprocessor is the
    escape hatch, so it is the escape hatch that is policed.
    """
    train_ids, val_ids, test_ids = folds
    leaky = cohort.fit_preprocessor(cohort.identifiers)
    dm = CohortDataModule(cohort, train_ids, val_ids, test_ids, batch_size=8, preprocessor=leaky)

    with pytest.raises(LeakageError) as excinfo:
        dm.setup()
    message = str(excinfo.value)
    assert "validation" in message
    assert val_ids[0] in message, "name the patients involved"


def test_a_legitimately_supplied_preprocessor_is_accepted(cohort, folds):
    """Reusing a fold's statistics downstream is a real workflow, not a mistake."""
    train_ids, _, test_ids = folds
    prep = cohort.fit_preprocessor(train_ids)
    dm = CohortDataModule(cohort, train_ids, test_ids=test_ids, batch_size=8, preprocessor=prep)
    dm.setup()

    assert dm.preprocessor is prep


def test_a_passthrough_preprocessor_never_trips_the_guard(make_cohort, folds):
    """It bakes in no statistics, so overlap with it means nothing."""
    cohort = make_cohort(
        continuous=["biomarker"],
        continuous_transform=None,
        categorical=[],
        categorical_transform=None,
    )
    train_ids, val_ids, test_ids = folds
    prep = cohort.fit_preprocessor(cohort.identifiers)
    CohortDataModule(cohort, train_ids, val_ids, test_ids, batch_size=8, preprocessor=prep).setup()


# =========================================================================== #
# loaders
# =========================================================================== #


def test_loaders_yield_patient_batches(cohort, folds):
    dm = CohortDataModule(cohort, *folds, batch_size=8)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, PatientBatch)
    assert batch.modalities["clinical"].shape[0] == 8
    assert batch.target["event"].dtype is torch.float32


def test_only_the_training_loader_shuffles(cohort, folds):
    """A shuffled validation loader makes per-batch metrics irreproducible."""
    train_ids, val_ids, test_ids = folds
    dm = CohortDataModule(cohort, train_ids, val_ids, test_ids, batch_size=8)
    dm.setup()

    assert dm.train_dataloader().sampler.__class__.__name__ == "RandomSampler"
    assert dm.val_dataloader().sampler.__class__.__name__ == "SequentialSampler"
    assert dm.test_dataloader().sampler.__class__.__name__ == "SequentialSampler"


def test_shuffling_can_be_turned_off(cohort, folds):
    dm = CohortDataModule(cohort, folds[0], batch_size=8, shuffle=False)
    dm.setup()
    assert dm.train_dataloader().sampler.__class__.__name__ == "SequentialSampler"


def test_no_validation_split_yields_an_empty_loader_not_none(cohort, folds):
    """Lightning 2.x rejects ``None`` from this hook but reads ``[]`` as "skip".

    Having no validation split is an ordinary thing to want and should not need a
    Trainer flag to express.
    """
    dm = CohortDataModule(cohort, folds[0], batch_size=8)
    dm.setup()
    assert dm.val_dataloader() == []


def test_asking_for_a_test_loader_without_a_test_split_raises(cohort, folds):
    """The opposite of validation, deliberately.

    ``Trainer.test()`` is an explicit request for a result; returning nothing
    would let it succeed having evaluated no rows.
    """
    dm = CohortDataModule(cohort, folds[0], batch_size=8)
    dm.setup()
    with pytest.raises(ValueError, match="no test split"):
        dm.test_dataloader()


def test_repr_reports_the_split_sizes_and_batch_size(cohort, folds):
    text = repr(CohortDataModule(cohort, *folds, batch_size="full"))
    assert "train=45" in text
    assert "batch_size='full'" in text


# =========================================================================== #
# Lightning integration
# =========================================================================== #


def test_a_trainer_can_fit_and_test_through_the_data_module(cohort, folds):
    """The end the whole module exists for: Lightning drives it with no adapters."""
    import pytorch_lightning as L

    class Toy(L.LightningModule):
        def __init__(self, n_features: int):
            super().__init__()
            self.net = torch.nn.Linear(n_features, 1)
            self.seen: list[str] = []

        def training_step(self, batch, _):
            return self.net(batch.modalities["clinical"]).mean().abs()

        def validation_step(self, batch, _):
            self.log("val_loss", self.net(batch.modalities["clinical"]).mean().abs())

        def test_step(self, batch, _):
            self.seen.extend(batch.patient_id)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    train_ids, val_ids, test_ids = folds
    dm = CohortDataModule(cohort, train_ids, val_ids, test_ids, batch_size=8)
    dm.setup()
    model = Toy(dm.preprocessor.width)

    kwargs = dict(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    L.Trainer(**kwargs).fit(model, datamodule=dm)
    L.Trainer(**kwargs).test(model, datamodule=dm, verbose=False)

    assert sorted(model.seen) == sorted(dm.test_ds.identifiers), "every test patient seen exactly once"
