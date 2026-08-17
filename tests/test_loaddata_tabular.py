"""Tests for ``kalecancer.loaddata.tabular.TabularDataset``.

Grouped by the design principle each set defends, because that is what makes a
failure here interpretable: if ``test_held_out_rows_use_train_statistics_exactly``
goes red, the package has a data leak, not a broken assertion.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from kalecancer.loaddata.base import NotFittedError
from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.survival.survival_target import SurvivalTarget
from tests.conftest import (
    CATEGORICAL,
    CONTINUOUS,
    EVENT_COLUMN,
    EVENT_VALUE,
    MISSING_AGE_ROWS,
    RARE_STAGE_ROWS,
    TIME_COLUMN,
    write_table,
)


def matrix_of(dataset: TabularDataset) -> np.ndarray:
    """The dataset's feature matrix, assembled through the public accessor only."""
    return torch.stack([dataset.get_by_id(i) for i in dataset.identifiers]).numpy()


def event_rate(dataset: TabularDataset) -> float:
    target = cast(SurvivalTarget, dataset.target)
    return float(target.events_for(dataset.identifiers).mean())


# =========================================================================== #
# reading the table
# =========================================================================== #


def test_every_supported_format_reads_identically(frame, tmp_path, any_format, make_target):
    path = write_table(frame, tmp_path, any_format)
    dataset = TabularDataset(path, identifier="patient_id", target=make_target())

    assert dataset.identifiers == frame["patient_id"].tolist()
    np.testing.assert_allclose(
        dataset.frame["biomarker"].to_numpy(dtype=float), frame["biomarker"].to_numpy(dtype=float)
    )


def test_leading_zeros_in_the_identifier_survive_every_reader(frame, tmp_path, any_format):
    """The bug this guards against is silent: ``"001"`` read as ``1`` joins nothing.

    ``pd.read_json`` and ``pd.read_csv`` both infer a zero-padded identifier as an
    integer unless told otherwise, and the damage only shows up much later, when a
    slide manifest keyed on ``"001"`` matches no rows at all.
    """
    path = write_table(frame, tmp_path, any_format)
    dataset = TabularDataset(path, identifier="patient_id")

    assert dataset.identifiers[0] == "001"
    assert all(isinstance(i, str) for i in dataset.identifiers)
    assert dataset.frame["patient_id"].tolist() == frame["patient_id"].tolist()


def test_naive_json_reading_would_have_lost_them(frame, tmp_path):
    """Pins the premise of the test above, so it cannot quietly stop being true."""
    path = write_table(frame, tmp_path, "json")
    assert pd.read_json(path)["patient_id"].iloc[0] == 1


@pytest.mark.parametrize("suffix", [".xlsx", ".parquet"])
def test_unsupported_format_raises_and_lists_the_supported_ones(tmp_path, suffix):
    """Parquet is included deliberately: dropping a reader must close the door on it.

    A format left half-supported -- listed in the message, unreadable in practice --
    is worse than one that was never offered.
    """
    path = tmp_path / f"cohort{suffix}"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported table format") as excinfo:
        TabularDataset(path, identifier="patient_id")
    assert ".csv" in str(excinfo.value)


def test_identifiers_follow_file_order(frame, table_path):
    assert TabularDataset(table_path, identifier="patient_id").identifiers == frame["patient_id"].tolist()


def test_duplicate_identifiers_raise(frame, tmp_path):
    doubled = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    path = write_table(doubled, tmp_path)
    with pytest.raises(ValueError, match="must be unique") as excinfo:
        TabularDataset(path, identifier="patient_id")
    assert "001" in str(excinfo.value), "the message should name the offenders"


# =========================================================================== #
# an in-memory frame as the source
#
# The seam that lets anything happen between reading the file and binding the
# target -- resolving rows the target refuses to guess at, joining a second
# table -- in the caller's script rather than in a separately-prepared file.
# =========================================================================== #


def test_a_frame_source_gives_the_same_dataset_as_the_file(frame, table_path, make_target):
    from_file = TabularDataset(table_path, identifier="patient_id", target=make_target())
    from_frame = TabularDataset(frame, identifier="patient_id", target=make_target())

    assert from_frame.identifiers == from_file.identifiers
    assert from_frame.frame.columns.tolist() == from_file.frame.columns.tolist()
    np.testing.assert_allclose(
        from_frame.frame["biomarker"].to_numpy(dtype=float),
        from_file.frame["biomarker"].to_numpy(dtype=float),
    )


def test_a_frame_source_leaves_path_unset(frame):
    """``path`` stays honest: there is no file behind this dataset."""
    assert TabularDataset(frame, identifier="patient_id").path is None


def test_the_callers_frame_is_copied_not_aliased(frame):
    """A dataset that shares its source frame would mutate under the caller's feet."""
    dataset = TabularDataset(frame, identifier="patient_id", continuous=["biomarker"])
    original = frame.loc[0, "biomarker"]

    frame.loc[0, "biomarker"] = 999.0

    assert dataset.frame.loc[0, "biomarker"] == original


def test_normalising_the_frame_does_not_mutate_the_callers_copy(frame):
    """The identifier is cast to string on the copy, not on what the caller holds."""
    frame["patient_id"] = np.arange(1, len(frame) + 1)

    dataset = TabularDataset(frame, identifier="patient_id")

    assert dataset.identifiers[0] == "1", "the dataset's own copy is normalised"
    assert frame["patient_id"].iloc[0] == 1, "the caller's frame is untouched"


def test_a_frame_source_is_reindexed_so_filtered_rows_do_not_misalign(frame, make_target):
    """The realistic entry point: a filtered frame arrives with gaps in its index.

    Without ``reset_index``, positional and label indexing diverge and ``subset``
    silently returns the wrong patients' rows.
    """
    filtered = frame[frame["stage"] != "I"]
    assert filtered.index.tolist() != list(range(len(filtered))), "fixture must have gaps"

    dataset = TabularDataset(filtered, identifier="patient_id", target=make_target())

    assert dataset.frame.index.tolist() == list(range(len(filtered)))
    assert dataset.identifiers == filtered["patient_id"].tolist()

    row = len(dataset) - 1
    item = dataset.frame.iloc[row]
    assert item["patient_id"] == dataset.identifiers[row]


def test_unknown_event_status_can_be_resolved_in_the_callers_script(frame, make_target):
    """End to end, the workflow the target's error message prescribes."""
    frame.loc[[4, 11], EVENT_COLUMN] = np.nan

    with pytest.raises(ValueError, match="missing value"):
        TabularDataset(frame, identifier="patient_id", target=make_target())

    resolved = frame[frame[EVENT_COLUMN].notna()]
    dataset = TabularDataset(resolved, identifier="patient_id", target=make_target())

    assert len(dataset) == len(frame) - 2
    assert frame.loc[4, "patient_id"] not in dataset


def test_a_frame_source_still_validates_everything_a_file_does(frame):
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        TabularDataset(doubled, identifier="patient_id")


# =========================================================================== #
# declaring columns
# =========================================================================== #


def test_missing_identifier_column_raises(table_path):
    with pytest.raises(ValueError, match="Identifier column 'nope' not found"):
        TabularDataset(table_path, identifier="nope")


def test_missing_feature_column_raises(table_path):
    with pytest.raises(ValueError, match=r"Feature column\(s\) \['nope'\] not found"):
        TabularDataset(table_path, identifier="patient_id", continuous=["nope"])


def test_a_column_cannot_be_both_continuous_and_categorical(table_path):
    with pytest.raises(ValueError, match="both continuous and categorical"):
        TabularDataset(table_path, identifier="patient_id", continuous=["age"], categorical=["age"])


def test_the_identifier_cannot_also_be_a_feature(table_path):
    with pytest.raises(ValueError, match="cannot also be a feature"):
        TabularDataset(table_path, identifier="patient_id", categorical=["patient_id"])


def test_feature_columns_are_continuous_then_categorical(cohort):
    assert cohort.feature_columns == CONTINUOUS + CATEGORICAL


def test_declared_columns_may_be_any_sequence(table_path):
    """Tuples are normalised to lists, so ``feature_columns`` can concatenate them."""
    dataset = TabularDataset(table_path, identifier="patient_id", continuous=("biomarker",), categorical=())
    assert dataset.continuous == ["biomarker"]
    assert dataset.categorical == []
    assert dataset.feature_columns == ["biomarker"]


# =========================================================================== #
# what you see is what you get -- no default preprocessing, ever
# =========================================================================== #


def test_a_transform_string_shorthand_is_refused(table_path):
    with pytest.raises(ValueError, match="no shorthand") as excinfo:
        TabularDataset(table_path, identifier="patient_id", continuous=["age"], continuous_transform="auto")
    assert "modelling decision" in str(excinfo.value)


def test_untransformed_categorical_columns_raise_rather_than_crash_later(make_dataset):
    with pytest.raises(ValueError, match="are not numeric and no categorical_transform") as excinfo:
        make_dataset(continuous=[], continuous_transform=None, categorical_transform=None)
    assert "OneHotEncoder" in str(excinfo.value), "the message should suggest the fix"


def test_untransformed_missing_values_raise_rather_than_reach_the_model(make_dataset):
    with pytest.raises(ValueError, match="no continuous_transform was declared to handle") as excinfo:
        make_dataset(categorical=[], categorical_transform=None, continuous_transform=None)
    assert "age" in str(excinfo.value)
    assert str(len(MISSING_AGE_ROWS)) in str(excinfo.value), "count the missing values"


def test_declaring_a_transform_for_one_role_does_not_clean_up_the_other(make_dataset):
    """The half-declared spec: scaled continuous columns beside raw string categoricals.

    This used to slip past the guard -- which only ran when *no* role declared a
    transform -- and die much later inside the numeric cast, as an opaque
    ``could not convert string to float: 'F'`` with no mention of which column or
    what to do about it.
    """
    with pytest.raises(ValueError, match="are not numeric and no categorical_transform") as excinfo:
        make_dataset(categorical_transform=None)

    message = str(excinfo.value)
    assert "could not convert string to float" not in message
    for column in CATEGORICAL:
        assert column in message, "name every offending column"
    assert "OneHotEncoder" in message


def test_the_half_declared_spec_fails_at_construction_not_at_fit_time(make_dataset):
    """Same loudness and same moment as a fully undeclared one: before any fold runs."""
    with pytest.raises(ValueError):
        make_dataset(continuous_transform=None, categorical_transform=None)
    with pytest.raises(ValueError):
        make_dataset(categorical_transform=None)


def test_missing_values_in_an_untransformed_categorical_role_are_caught(make_dataset):
    """And the suggested imputer is the one that suits the role."""
    with pytest.raises(ValueError, match="no categorical_transform was declared to handle") as excinfo:
        make_dataset(
            continuous=[],
            continuous_transform=None,
            categorical=["age"],
            categorical_transform=None,
        )
    assert "most_frequent" in str(excinfo.value), "median makes no sense for a categorical"


def test_a_non_numeric_column_declared_as_continuous_is_caught(make_dataset):
    """A likely mis-declaration, so the message points at the role, not just a transform."""
    with pytest.raises(ValueError, match="Continuous column") as excinfo:
        make_dataset(continuous=["sex"], continuous_transform=None, categorical=[], categorical_transform=None)
    assert "as categorical instead" in str(excinfo.value)


def test_clean_numeric_columns_with_no_transform_pass_through_untouched(frame, make_dataset):
    """The other half of WYSIWYG: nothing declared means nothing done."""
    dataset = make_dataset(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    assert dataset.is_fitted, "nothing to fit, so values are available immediately"
    np.testing.assert_allclose(matrix_of(dataset).ravel(), frame["biomarker"].to_numpy(dtype=float), rtol=1e-6)


def test_declaring_no_features_at_all_is_allowed(make_dataset):
    dataset = make_dataset(continuous=[], continuous_transform=None, categorical=[], categorical_transform=None)
    assert dataset.n_features == 0
    assert dataset.get_by_id("001").shape == (0,)


def test_one_role_may_pass_through_while_the_other_is_transformed(frame, make_dataset):
    """A clean numeric column can sit untouched next to encoded categoricals.

    Worth pinning because it is the boundary of the guard above: declaring a
    transform for only one role is fine *as long as* the passed-through columns
    are already numeric and complete. Partial declaration is not itself the
    problem, so the guard must not over-fire on this.
    """
    dataset = make_dataset(
        continuous=["biomarker"],
        continuous_transform=None,
        categorical=["sex"],
        categorical_transform=OneHotEncoder(sparse_output=False),
    )
    assert dataset.is_fitted is False, "the categorical role still has something to fit"

    fitted = dataset.fit_transform()
    assert fitted.feature_names == ["biomarker", "sex_female", "sex_male"]
    np.testing.assert_allclose(matrix_of(fitted)[:, 0], frame["biomarker"].to_numpy(dtype=float), rtol=1e-6)


def test_the_unfitted_message_names_only_what_is_actually_pending(make_dataset):
    """A passed-through role has nothing to fit and must not appear in the message."""
    dataset = make_dataset(
        continuous=["biomarker"],
        continuous_transform=None,
        categorical=["sex"],
        categorical_transform=OneHotEncoder(sparse_output=False),
    )
    with pytest.raises(NotFittedError) as excinfo:
        dataset[0]

    message = str(excinfo.value)
    assert "1 unfitted transform(s)" in message
    assert "OneHotEncoder" in message
    assert "passthrough" not in message


# =========================================================================== #
# fitted-state discipline
# =========================================================================== #


def test_construction_fits_nothing(cohort):
    assert cohort.is_fitted is False


def test_unfitted_dataset_refuses_to_serve_values(cohort):
    with pytest.raises(NotFittedError) as excinfo:
        cohort[0]
    message = str(excinfo.value)
    assert "SimpleImputer" in message and "StandardScaler" in message, "name what is pending"
    assert "fit_transform" in message, "and say how to fix it"


def test_the_exploration_hatch_works_while_unfitted(cohort, frame):
    """``.frame`` is how you look at your data before deciding how to treat it."""
    assert len(cohort.frame) == len(frame)
    assert cohort.frame["age"].isna().sum() == len(MISSING_AGE_ROWS)


def test_fit_transform_returns_a_new_instance_and_leaves_the_original_unfitted(cohort):
    fitted = cohort.fit_transform()

    assert fitted is not cohort
    assert fitted.is_fitted is True
    assert cohort.is_fitted is False
    with pytest.raises(NotFittedError):
        cohort[0]


def test_folds_fitted_from_one_parent_are_independent(cohort):
    """What makes parallel cross-validation and nested CV safe.

    Fitting a second fold must not disturb the first, and neither may reach back
    into the shared parent -- which stays unfitted throughout.
    """
    fold_a = cohort.subset(range(0, 40)).fit_transform()
    before = matrix_of(fold_a).copy()

    fold_b = cohort.subset(range(40, 80)).fit_transform()

    np.testing.assert_array_equal(matrix_of(fold_a), before)
    assert cohort.is_fitted is False

    # Each fold standardised against its own rows, so both centre on zero, and
    # each reproduces the pipeline fitted on precisely its own slice.
    for fold, rows in ((fold_a, range(0, 40)), (fold_b, range(40, 80))):
        reference = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        expected = reference.fit_transform(cohort.frame.iloc[list(rows)][CONTINUOUS])
        np.testing.assert_allclose(matrix_of(fold)[:, : len(CONTINUOUS)], expected, rtol=1e-5, atol=1e-6)


def test_transform_before_fitting_raises(cohort):
    train, test = cohort.split(test_size=0.25, random_state=0)
    with pytest.raises(NotFittedError, match="call fit_transform"):
        train.transform(test)


def test_transform_does_not_mutate_the_held_out_dataset(cohort):
    train, test = cohort.split(test_size=0.25, random_state=0)
    fitted_test = train.fit_transform().transform(test)

    assert fitted_test is not test
    assert fitted_test.is_fitted is True
    assert test.is_fitted is False, "the caller's object is untouched"


def test_a_dataset_with_no_transforms_is_fitted_from_the_start(make_dataset):
    """Nothing to fit means values are available immediately, and transform is a no-op."""
    dataset = make_dataset(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    train, test = dataset.split(test_size=0.25, random_state=0)

    assert dataset.is_fitted is True
    np.testing.assert_array_equal(matrix_of(train.transform(test)), matrix_of(test))


# =========================================================================== #
# leakage -- the property the whole clone-per-fold design exists to guarantee
# =========================================================================== #


def test_held_out_rows_use_train_statistics_exactly(cohort):
    """The strong form: the held-out matrix equals the train pipeline applied to it.

    Not "the test mean is nonzero" -- that is only evidence. This reproduces the
    declared pipeline by hand, fitted on exactly the training rows, and demands
    an exact match. Any statistic computed from the test rows breaks it.
    """
    train, test = cohort.split(test_size=0.25, random_state=0)
    fitted_train = train.fit_transform()
    fitted_test = fitted_train.transform(test)

    assert fitted_train.feature_names[: len(CONTINUOUS)] == CONTINUOUS

    reference = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    reference.fit(train.frame[CONTINUOUS])

    np.testing.assert_allclose(
        matrix_of(fitted_test)[:, : len(CONTINUOUS)],
        reference.transform(test.frame[CONTINUOUS]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_standardised_test_mean_is_not_zero(cohort):
    """The cheap smoke signal, kept because it is the one a human can eyeball.

    Train standardises to a mean of exactly zero by construction. If the held-out
    half also reads zero, its own statistics were used to standardise it.
    """
    train, test = cohort.split(test_size=0.25, random_state=0)
    fitted_train = train.fit_transform()
    fitted_test = fitted_train.transform(test)

    assert matrix_of(fitted_train)[:, 0].mean() == pytest.approx(0.0, abs=1e-5)
    assert abs(matrix_of(fitted_test)[:, 0].mean()) > 1e-4


def test_a_held_out_missing_value_is_filled_with_the_training_median(cohort):
    """A test row's own column must not contribute to the value used to fill it.

    The fold boundary is chosen explicitly rather than drawn, so the rows with a
    missing ``age`` are guaranteed to land in the held-out half.
    """
    train_rows = [i for i in range(len(cohort)) if i not in MISSING_AGE_ROWS]
    fitted_train = cohort.subset(train_rows).fit_transform()
    held_out = fitted_train.transform(cohort.subset(MISSING_AGE_ROWS))

    train_median = cohort.frame.iloc[train_rows]["age"].median()
    reference = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    reference.fit(cohort.frame.iloc[train_rows][CONTINUOUS])
    scaled_median = (train_median - reference[-1].mean_[0]) / reference[-1].scale_[0]

    for identifier in held_out.identifiers:
        assert held_out.get_by_id(identifier)[0].item() == pytest.approx(scaled_median, abs=1e-5)


def test_split_on_a_fitted_dataset_raises(cohort):
    """Both halves would inherit statistics fitted across all rows. Silently."""
    fitted = cohort.fit_transform()
    with pytest.raises(RuntimeError, match="leaking the held-out half"):
        fitted.split(test_size=0.25, random_state=0)


def test_an_unseen_category_at_test_time_does_not_crash(cohort, frame):
    """``handle_unknown="ignore"`` on a small cohort: this will happen, not might.

    The fixture confines ``stage="IV"`` to the last two rows, so fitting on
    everything else guarantees the encoder meets it for the first time at
    transform time.
    """
    train_rows = [i for i in range(len(cohort)) if i not in RARE_STAGE_ROWS]
    fitted_train = cohort.subset(train_rows).fit_transform()
    held_out = fitted_train.transform(cohort.subset(RARE_STAGE_ROWS))

    assert "stage_IV" not in fitted_train.feature_names
    stage_columns = [i for i, n in enumerate(fitted_train.feature_names) if n.startswith("stage_")]
    encoded = matrix_of(held_out)[:, stage_columns]
    np.testing.assert_array_equal(encoded, np.zeros_like(encoded))


# =========================================================================== #
# splitting
# =========================================================================== #


def test_split_is_disjoint_and_exhaustive(cohort):
    train, test = cohort.split(test_size=0.25, random_state=0)

    assert len(train) + len(test) == len(cohort)
    assert set(train.identifiers).isdisjoint(test.identifiers)
    assert set(train.identifiers) | set(test.identifiers) == set(cohort.identifiers)


def test_split_honours_test_size(cohort):
    _, test = cohort.split(test_size=0.25, random_state=0)
    assert len(test) == round(0.25 * len(cohort))


def test_split_is_stratified_on_event_status(cohort):
    """At a few hundred patients an unstratified split skews the event rate badly.

    Checked over many seeds on purpose. Any single unstratified split has a fair
    chance of looking balanced, so a one-seed assertion cannot tell stratified
    from lucky -- across 20 seeds the distinction is unambiguous. Empirically the
    stratified worst case here is a 0.013 deviation; unstratified reaches 0.19.
    """
    overall = event_rate(cohort)
    deviations = []
    for seed in range(20):
        train, test = cohort.split(test_size=0.25, random_state=seed)
        deviations.append(abs(event_rate(train) - overall))
        deviations.append(abs(event_rate(test) - overall))

    assert max(deviations) < 0.02, (
        f"worst event-rate deviation {max(deviations):.3f} over 20 seeds; stratification looks to have been dropped"
    )


def test_split_is_reproducible(cohort):
    a, _ = cohort.split(test_size=0.25, random_state=7)
    b, _ = cohort.split(test_size=0.25, random_state=7)
    c, _ = cohort.split(test_size=0.25, random_state=8)

    assert a.identifiers == b.identifiers
    assert a.identifiers != c.identifiers


def test_both_halves_of_a_split_are_unfitted(cohort):
    train, test = cohort.split(test_size=0.25, random_state=0)
    assert train.is_fitted is False
    assert test.is_fitted is False


def test_split_works_without_a_target(make_dataset):
    dataset = make_dataset(target=None)
    train, test = dataset.split(test_size=0.25, random_state=0)
    assert len(train) + len(test) == len(dataset)


def test_split_keeps_frame_and_identifiers_aligned(cohort):
    """The failure this prevents is a patient's covariates paired to another's outcome."""
    train, test = cohort.split(test_size=0.25, random_state=0)
    for part in (train, test):
        assert part.frame["patient_id"].tolist() == part.identifiers


# =========================================================================== #
# identifier keying, end to end
# =========================================================================== #


def test_subsetting_a_fitted_dataset_keeps_each_patients_own_row(cohort):
    """Principle 4, and the one that produces meaningless results when broken.

    A composite reaches into components by identifier precisely so that a subset,
    a reorder or a fold boundary can never re-pair features with the wrong patient.
    """
    fitted = cohort.fit_transform()
    scattered = [11, 2, 79, 40, 0]
    sub = fitted.subset(scattered)

    assert sub.identifiers == [cohort.identifiers[i] for i in scattered]
    for identifier in sub.identifiers:
        torch.testing.assert_close(sub.get_by_id(identifier), fitted.get_by_id(identifier))


def test_get_by_id_rejects_an_unknown_identifier(cohort):
    fitted = cohort.fit_transform()
    with pytest.raises(KeyError):
        fitted.get_by_id("no-such-patient")


def test_membership_follows_the_subset(cohort):
    fitted = cohort.fit_transform()
    sub = fitted.subset([0, 1])
    assert "001" in sub
    assert "050" in fitted
    assert "050" not in sub


# =========================================================================== #
# the item contract
# =========================================================================== #


def test_item_dict_has_exactly_the_documented_keys(cohort):
    fitted = cohort.fit_transform()
    assert set(fitted[0]) == {"clinical", "patient_id", "time", "event"}


def test_feature_tensor_is_float32_and_one_dimensional(cohort):
    fitted = cohort.fit_transform()
    features = fitted[0]["clinical"]

    assert features.dtype is torch.float32
    assert features.shape == (fitted.n_features,)


def test_the_modality_name_is_the_feature_key(make_dataset):
    dataset = make_dataset(name="covariates")
    fitted = dataset.fit_transform()
    assert "covariates" in fitted[0]
    assert "clinical" not in fitted[0]


def test_default_modality_name_is_clinical(cohort):
    assert cohort.name == "clinical"


def test_item_matches_the_row_it_claims(cohort, frame):
    fitted = cohort.fit_transform()
    item = fitted[5]
    row = frame.loc[frame["patient_id"] == item["patient_id"]].iloc[0]

    assert item["time"].item() == pytest.approx(float(row[TIME_COLUMN]))
    assert item["event"].item() == pytest.approx(float(row[EVENT_COLUMN] == EVENT_VALUE))


def test_dataloader_collates_without_a_custom_collate_fn(cohort):
    fitted = cohort.fit_transform()
    batch = next(iter(DataLoader(fitted, batch_size=8, shuffle=False)))

    assert batch["clinical"].shape == (8, fitted.n_features)
    assert batch["time"].shape == (8,)
    assert batch["event"].shape == (8,)
    assert batch["patient_id"] == fitted.identifiers[:8], "strings come back as a list"


# =========================================================================== #
# encoding and reporting
# =========================================================================== #


def test_n_features_counts_declared_columns_before_fitting_and_encoded_after(cohort):
    assert cohort.n_features == len(CONTINUOUS + CATEGORICAL)

    fitted = cohort.fit_transform()
    assert fitted.n_features > cohort.n_features, "one-hot expands the categoricals"
    assert fitted.n_features == len(fitted.feature_names)
    assert fitted.n_features == matrix_of(fitted).shape[1]


def test_feature_names_are_readable(cohort):
    fitted = cohort.fit_transform()
    assert fitted.feature_names[: len(CONTINUOUS)] == CONTINUOUS
    assert "sex_male" in fitted.feature_names
    assert all("__" not in name for name in fitted.feature_names), "no transformer prefixes"


def test_feature_names_are_empty_until_fitted(cohort):
    assert cohort.feature_names == []


def test_describe_transforms_reports_the_chain_and_the_status(cohort):
    description = cohort.describe_transforms()

    assert "SimpleImputer -> StandardScaler" in description
    assert "SimpleImputer -> OneHotEncoder" in description
    assert "unfitted - fitted per fold" in description

    assert "fitted" in cohort.fit_transform().describe_transforms()


def test_describe_transforms_says_when_nothing_is_declared(make_dataset):
    dataset = make_dataset(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    description = dataset.describe_transforms()
    assert "passthrough" in description
    assert "none declared" in description


def test_repr_surfaces_the_event_rate(cohort, frame):
    """So a mis-mapped ``event_value`` is visible the moment you print the cohort."""
    n_events = int(frame[EVENT_COLUMN].eq(EVENT_VALUE).sum())
    text = repr(cohort)

    assert "TabularDataset(" in text
    assert f"{len(cohort)} samples" in text
    assert f"{n_events} events" in text
    assert "unfitted" in text


def test_repr_reports_encoded_width_once_fitted(cohort):
    fitted = cohort.fit_transform()
    text = repr(fitted)
    assert f"{fitted.n_features} features" in text
    assert "unfitted" not in text


def test_repr_distinguishes_nothing_to_fit_from_not_yet_fitted(make_dataset):
    """``unfitted`` must not cover both: one of these serves values, the other raises.

    Reporting them alike also contradicts ``is_fitted``, which is ``True`` here --
    the state a reader is most likely to check the repr for in the first place.
    """
    passthrough = make_dataset(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    text = repr(passthrough)

    assert "no transforms" in text
    assert "unfitted" not in text
    assert passthrough.is_fitted


def test_repr_without_a_target_omits_survival_summary(make_dataset):
    assert "events" not in repr(make_dataset(target=None))


# =========================================================================== #
# target integration
# =========================================================================== #


def test_the_target_is_bound_at_construction(cohort, frame):
    n_events = int(frame[EVENT_COLUMN].eq(EVENT_VALUE).sum())
    assert cohort.target.events_for(cohort.identifiers).sum() == n_events


def test_a_mis_specified_event_value_fails_at_construction(table_path, make_target):
    """Not at training time, and not as a c-index of 0.31 three hours later."""
    with pytest.raises(ValueError, match="matches 0%"):
        TabularDataset(table_path, identifier="patient_id", target=make_target(event_value="dead"))


def test_a_target_over_a_missing_column_fails_at_construction(table_path):
    with pytest.raises(ValueError, match="not found"):
        TabularDataset(
            table_path,
            identifier="patient_id",
            target=SurvivalTarget(time="no_such_column", event=EVENT_COLUMN, event_value=EVENT_VALUE),
        )


def test_columns_used_as_supervision_are_not_silently_also_features(cohort):
    """Declaring the outcome as a covariate would be a perfect, useless predictor.

    Nothing stops a caller doing it deliberately, but it must never happen by
    default: only explicitly declared columns become features.
    """
    fitted = cohort.fit_transform()
    assert TIME_COLUMN not in fitted.feature_names
    assert EVENT_COLUMN not in fitted.feature_names
    assert not any(name.startswith(EVENT_COLUMN) for name in fitted.feature_names)


# The half-declared spec that used to escape the guard is covered above, under
# "what you see is what you get": see
# test_declaring_a_transform_for_one_role_does_not_clean_up_the_other.
