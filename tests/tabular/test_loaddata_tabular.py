"""``TabularCohort``: reading, validating, fitting per fold, and serving samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from kalecancer.loaddata.base import NotFittedError
from kalecancer.loaddata.sample import collate_samples
from kalecancer.loaddata.tabular import TabularCohort
from kalecancer.survival.survival_target import SurvivalTarget
from tests.tabular.conftest import (
    CATEGORICAL,
    CONTINUOUS,
    EVENT_COLUMN,
    EVENT_VALUE,
    MISSING_AGE_ROWS,
    RARE_STAGE_ROWS,
    TIME_COLUMN,
    event_rate,
    fitted_view,
    ids_at,
    matrix_of,
    write_table,
)


def names(view) -> list[str]:
    return view.feature_names["clinical"]


# =========================================================================== #
# reading
# =========================================================================== #


def test_every_supported_format_reads_identically(frame, tmp_path, any_format, make_target):
    path = write_table(frame, tmp_path, any_format)
    cohort = TabularCohort(path, identifier="patient_id", target=make_target())
    assert cohort.identifiers == frame["patient_id"].tolist()


def test_leading_zeros_in_the_identifier_survive_every_reader(frame, tmp_path, any_format):
    """``"001"`` inferred as ``1`` joins against a slide manifest and matches nothing."""
    path = write_table(frame, tmp_path, any_format)
    cohort = TabularCohort(path, identifier="patient_id")
    assert cohort.identifiers[0] == "001"
    assert "001" in cohort


def test_unsupported_format_raises_and_lists_the_supported_ones(tmp_path):
    path = tmp_path / "cohort.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"\.csv"):
        TabularCohort(path, identifier="patient_id")


def test_identifiers_follow_file_order(frame, table_path):
    assert TabularCohort(table_path, identifier="patient_id").identifiers == frame["patient_id"].tolist()


def test_duplicate_identifiers_raise(frame, tmp_path):
    frame.loc[1, "patient_id"] = frame.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="must be unique"):
        TabularCohort(write_table(frame, tmp_path), identifier="patient_id")


def test_a_frame_source_gives_the_same_cohort_as_the_file(frame, table_path, make_target):
    from_file = TabularCohort(table_path, identifier="patient_id", target=make_target())
    from_frame = TabularCohort(frame, identifier="patient_id", target=make_target())
    assert from_frame.identifiers == from_file.identifiers
    assert from_frame.path is None


def test_the_callers_frame_is_copied_not_aliased(frame):
    cohort = TabularCohort(frame, identifier="patient_id")
    frame.loc[0, "biomarker"] = -999.0
    assert cohort.frame.loc[0, "biomarker"] != -999.0


def test_a_frame_source_is_reindexed_so_filtered_rows_do_not_misalign(frame, make_target):
    """The seam for resolving rows the target refuses to guess at.

    A filtered frame keeps its original index; not resetting it would pair row 0's
    covariates with row 7's outcome.
    """
    filtered = frame[frame[EVENT_COLUMN].notna()].iloc[10:]
    cohort = TabularCohort(filtered, identifier="patient_id", target=make_target())
    assert cohort.frame.index.tolist() == list(range(len(filtered)))
    assert cohort.identifiers == filtered["patient_id"].tolist()


# =========================================================================== #
# declaration -- validated at construction, never later
# =========================================================================== #


def test_missing_identifier_column_raises(table_path):
    with pytest.raises(ValueError, match="Identifier column"):
        TabularCohort(table_path, identifier="nope")


def test_missing_feature_column_raises(table_path):
    with pytest.raises(ValueError, match="Feature column"):
        TabularCohort(table_path, identifier="patient_id", continuous=["nope"])


def test_a_column_cannot_be_both_continuous_and_categorical(table_path):
    with pytest.raises(ValueError, match="both continuous and categorical"):
        TabularCohort(table_path, identifier="patient_id", continuous=["age"], categorical=["age"])


def test_the_identifier_cannot_also_be_a_feature(table_path):
    with pytest.raises(ValueError, match="cannot also be a feature"):
        TabularCohort(table_path, identifier="patient_id", categorical=["patient_id"])


def test_feature_columns_are_continuous_then_categorical(cohort):
    assert cohort.feature_columns == CONTINUOUS + CATEGORICAL


def test_a_transform_string_shorthand_is_refused(table_path):
    """A shorthand would make the preprocessing invisible in the user's script."""
    with pytest.raises(ValueError, match="not the string"):
        TabularCohort(table_path, identifier="patient_id", continuous=["age"], continuous_transform="standard")


# =========================================================================== #
# what you see is what you get
# =========================================================================== #


def test_untransformed_categorical_columns_raise_rather_than_crash_later(make_cohort):
    with pytest.raises(ValueError, match="are not numeric"):
        make_cohort(categorical_transform=None)


def test_untransformed_missing_values_raise_rather_than_reach_the_model(make_cohort):
    with pytest.raises(ValueError, match="Missing values"):
        make_cohort(continuous_transform=None, categorical=[], categorical_transform=None)


def test_declaring_a_transform_for_one_role_does_not_clean_up_the_other(make_cohort):
    """The half-declared spec: scaled numbers beside raw strings.

    Checking the feature block as a whole would let this through, and it would die
    much later inside the numeric cast as ``could not convert string to float``.
    """
    with pytest.raises(ValueError, match="Categorical column"):
        make_cohort(categorical_transform=None)


def test_a_non_numeric_column_declared_as_continuous_is_caught(make_cohort):
    with pytest.raises(ValueError, match="are not numeric"):
        make_cohort(continuous=["sex"], continuous_transform=None, categorical=[], categorical_transform=None)


def test_clean_numeric_columns_with_no_transform_pass_through_untouched(frame, make_cohort):
    cohort = make_cohort(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    view = fitted_view(cohort)
    assert names(view) == ["biomarker"]
    np.testing.assert_allclose(matrix_of(view)[:, 0], frame["biomarker"].to_numpy(dtype=float), rtol=1e-6)


def test_declaring_no_features_at_all_is_allowed(make_cohort):
    """A cohort can be a pure target carrier for a multimodal composite."""
    cohort = make_cohort(continuous=[], continuous_transform=None, categorical=[], categorical_transform=None)
    assert cohort.feature_columns == []


def test_one_role_may_pass_through_while_the_other_is_transformed(frame, make_cohort):
    cohort = make_cohort(
        continuous=["biomarker"],
        continuous_transform=None,
        categorical=["sex"],
        categorical_transform=OneHotEncoder(sparse_output=False),
    )
    view = fitted_view(cohort)
    assert names(view) == ["biomarker", "sex_female", "sex_male"]
    np.testing.assert_allclose(matrix_of(view)[:, 0], frame["biomarker"].to_numpy(dtype=float), rtol=1e-6)


# =========================================================================== #
# fitted state lives outside the cohort
# =========================================================================== #


def test_construction_fits_nothing(cohort):
    """A cohort is never fitted. That is the whole point of the split."""
    assert not hasattr(cohort, "_preprocessor")


def test_building_a_view_without_a_preprocessor_refuses(cohort):
    """Refused at view construction, not at first access.

    A tabular cohort's ``fit_preprocessor`` always returns an artifact -- a
    passthrough is still an artifact -- so ``None`` here is always a mistake, and
    the earliest place to say so is when the view is built.
    """
    with pytest.raises(NotFittedError) as excinfo:
        cohort.view(ids_at(cohort, range(4)), None)
    message = str(excinfo.value)
    assert "fit_preprocessor" in message, "say how to fix it"
    assert "cohort.frame" in message, "and where the raw values are"


def test_the_exploration_hatch_works_before_any_fitting(cohort, frame):
    assert len(cohort.frame) == len(frame)
    assert cohort.frame["age"].isna().sum() == len(MISSING_AGE_ROWS)


def test_fitting_does_not_mutate_the_cohort(cohort):
    """What makes parallel cross-validation and nested CV safe."""
    before = cohort.frame.copy()
    cohort.fit_preprocessor(ids_at(cohort, range(40)))
    cohort.fit_preprocessor(ids_at(cohort, range(40, 80)))
    pd.testing.assert_frame_equal(cohort.frame, before)


def test_folds_fitted_from_one_cohort_are_independent(cohort):
    """Each fold standardises against its own rows and nothing reaches across."""
    fold_a = fitted_view(cohort, ids_at(cohort, range(0, 40)))
    before = matrix_of(fold_a).copy()

    fold_b = fitted_view(cohort, ids_at(cohort, range(40, 80)))
    np.testing.assert_array_equal(matrix_of(fold_a), before)

    for fold, rows in ((fold_a, range(0, 40)), (fold_b, range(40, 80))):
        reference = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        expected = reference.fit_transform(cohort.frame.iloc[list(rows)][CONTINUOUS])
        np.testing.assert_allclose(matrix_of(fold)[:, : len(CONTINUOUS)], expected, rtol=1e-5, atol=1e-6)


def test_a_preprocessor_records_the_rows_it_was_fitted_on(cohort):
    """Provenance by identifier, which is what makes the leakage check possible."""
    prep = cohort.fit_preprocessor(ids_at(cohort, range(0, 40)))
    assert prep.fitted_on == frozenset(cohort.identifiers[:40])


def test_a_passthrough_preprocessor_claims_no_rows(make_cohort):
    """It bakes in no statistics, so no row's information is inside it.

    An empty ``fitted_on`` is the honest answer, and it keeps the leakage guard
    from firing on a configuration that cannot leak.
    """
    cohort = make_cohort(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    assert cohort.fit_preprocessor(ids_at(cohort, range(40))).fitted_on == frozenset()


def test_the_same_preprocessor_serves_every_view_of_a_fold(cohort):
    prep = cohort.fit_preprocessor(ids_at(cohort, range(60)))
    train, held_out = cohort.view(ids_at(cohort, range(60)), prep), cohort.view(ids_at(cohort, range(60, 80)), prep)
    assert train.preprocessor is prep and held_out.preprocessor is prep


# =========================================================================== #
# leakage -- the property the design exists to guarantee
# =========================================================================== #


def test_held_out_rows_use_train_statistics_exactly(cohort):
    """The strong form: the held-out matrix equals the train pipeline applied to it.

    Not "the test mean is nonzero" -- that is only evidence. This reproduces the
    declared pipeline by hand, fitted on exactly the training rows, and demands an
    exact match. Any statistic computed from the test rows breaks it.
    """
    train_ids, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=True)
    prep = cohort.fit_preprocessor(train_ids)
    test_view = cohort.view(test_ids, prep)

    assert prep.feature_names["clinical"][: len(CONTINUOUS)] == CONTINUOUS

    frame = cohort.frame.set_index("patient_id")
    reference = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    reference.fit(frame.loc[train_ids, CONTINUOUS])

    np.testing.assert_allclose(
        matrix_of(test_view)[:, : len(CONTINUOUS)],
        reference.transform(frame.loc[test_ids, CONTINUOUS]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_standardised_test_mean_is_not_zero(cohort):
    """The cheap smoke signal, kept because it is the one a human can eyeball."""
    train_ids, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=True)
    prep = cohort.fit_preprocessor(train_ids)

    assert matrix_of(cohort.view(train_ids, prep))[:, 0].mean() == pytest.approx(0.0, abs=1e-5)
    assert abs(matrix_of(cohort.view(test_ids, prep))[:, 0].mean()) > 1e-4


def test_a_held_out_missing_value_is_filled_with_the_training_median(cohort):
    """A test row's own column must not contribute to the value used to fill it."""
    train_rows = [i for i in range(len(cohort)) if i not in MISSING_AGE_ROWS]
    prep = cohort.fit_preprocessor(ids_at(cohort, train_rows))
    held_out = cohort.view(ids_at(cohort, MISSING_AGE_ROWS), prep)

    train_median = cohort.frame.iloc[train_rows]["age"].median()
    reference = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    reference.fit(cohort.frame.iloc[train_rows][CONTINUOUS])
    scaled_median = (train_median - reference[-1].mean_[0]) / reference[-1].scale_[0]

    for i in range(len(held_out)):
        assert held_out[i].modalities["clinical"][0].item() == pytest.approx(scaled_median, abs=1e-5)


def test_an_unseen_category_at_test_time_does_not_crash(cohort):
    """``handle_unknown="ignore"`` on a small cohort: this will happen, not might."""
    train_rows = [i for i in range(len(cohort)) if i not in RARE_STAGE_ROWS]
    prep = cohort.fit_preprocessor(ids_at(cohort, train_rows))
    held_out = cohort.view(ids_at(cohort, RARE_STAGE_ROWS), prep)

    encoded_names = prep.feature_names["clinical"]
    assert "stage_IV" not in encoded_names
    stage_columns = [i for i, n in enumerate(encoded_names) if n.startswith("stage_")]
    encoded = matrix_of(held_out)[:, stage_columns]
    np.testing.assert_array_equal(encoded, np.zeros_like(encoded))


def test_a_cohort_cannot_be_split_into_a_fitted_state(cohort):
    """The ``split() after fit`` trap cannot be expressed: ``split`` returns indices and
    a cohort holds no fitted state, so neither half inherits anything."""
    cohort.fit_preprocessor(cohort.identifiers)
    train_ids, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=True)
    assert set(train_ids).isdisjoint(test_ids)


# =========================================================================== #
# splitting
# =========================================================================== #


def test_split_is_disjoint_and_exhaustive(cohort):
    train_ids, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=True)
    assert len(train_ids) + len(test_ids) == len(cohort)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(cohort.identifiers)


def test_split_honours_test_size(cohort):
    _, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=True)
    assert len(test_ids) == round(0.25 * len(cohort))


def test_split_is_stratified_on_event_status(cohort):
    """At a few hundred patients an unstratified split skews the event rate badly.

    Checked over many seeds on purpose. Any single unstratified split has a fair
    chance of looking balanced, so a one-seed assertion cannot tell stratified
    from lucky -- across 20 seeds the distinction is unambiguous.
    """
    overall = event_rate(cohort, cohort.identifiers)
    deviations = []
    for seed in range(20):
        train_ids, test_ids = cohort.split(test_size=0.25, random_state=seed, stratify=True)
        for part in (train_ids, test_ids):
            deviations.append(abs(event_rate(cohort, part) - overall))

    assert max(deviations) < 0.02, (
        f"worst event-rate deviation {max(deviations):.3f} over 20 seeds; stratification looks to have been dropped"
    )


def test_split_is_reproducible(cohort):
    a, _ = cohort.split(test_size=0.25, random_state=7, stratify=True)
    b, _ = cohort.split(test_size=0.25, random_state=7, stratify=True)
    c, _ = cohort.split(test_size=0.25, random_state=8, stratify=True)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_split_works_without_a_target_when_stratification_is_declined(make_cohort):
    cohort = make_cohort(target=None)
    train_ids, test_ids = cohort.split(test_size=0.25, random_state=0, stratify=False)
    assert len(train_ids) + len(test_ids) == len(cohort)


# =========================================================================== #
# identifier keying, end to end
# =========================================================================== #


def test_a_scattered_view_keeps_each_patients_own_row(cohort):
    """Rule 4, and the one that produces meaningless results when broken."""
    prep = cohort.fit_preprocessor(cohort.identifiers)
    scattered = ids_at(cohort, [11, 2, 79, 40, 0])
    view = cohort.view(scattered, prep)

    assert view.identifiers == scattered
    for position, identifier in enumerate(scattered):
        torch.testing.assert_close(view[position].modalities["clinical"], cohort.payload(identifier, prep)["clinical"])


def test_payload_rejects_an_unknown_identifier(cohort):
    prep = cohort.fit_preprocessor(cohort.identifiers)
    with pytest.raises(KeyError):
        cohort.payload("no-such-patient", prep)


def test_the_bulk_and_per_sample_paths_agree(cohort):
    """A view caches through ``payload_bulk``; both must give identical answers."""
    prep = cohort.fit_preprocessor(cohort.identifiers)
    view = cohort.view(cohort.identifiers, prep)

    for i in (0, 17, 79):
        torch.testing.assert_close(
            view[i].modalities["clinical"], cohort.payload(cohort.identifiers[i], prep)["clinical"]
        )


# =========================================================================== #
# the item contract
# =========================================================================== #


def test_a_sample_carries_exactly_the_documented_fields(cohort):
    item = fitted_view(cohort)[0]
    assert set(item.modalities) == {"clinical"}
    assert set(item.present) == {"clinical"}
    assert set(item.target) == {"time", "event"}
    assert item.patient_id == cohort.identifiers[0]


def test_feature_tensor_is_float32_and_one_dimensional(cohort):
    view = fitted_view(cohort)
    features = view[0].modalities["clinical"]
    assert features.dtype is torch.float32
    assert features.shape == (len(names(view)),)


def test_the_modality_name_is_the_feature_key(make_cohort):
    view = fitted_view(make_cohort(name="covariates"))
    assert set(view[0].modalities) == {"covariates"}


def test_default_modality_name_is_clinical(cohort):
    assert cohort.name == "clinical"


def test_a_sample_matches_the_row_it_claims(cohort, frame):
    item = fitted_view(cohort)[5]
    row = frame.loc[frame["patient_id"] == item.patient_id].iloc[0]

    assert item.target["time"].item() == pytest.approx(float(row[TIME_COLUMN]))
    assert item.target["event"].item() == pytest.approx(float(row[EVENT_COLUMN] == EVENT_VALUE))


def test_a_dataloader_batches_a_view(cohort):
    view = fitted_view(cohort)
    batch = next(iter(DataLoader(view, batch_size=8, shuffle=False, collate_fn=collate_samples)))

    assert batch.modalities["clinical"].shape == (8, len(names(view)))
    assert batch.target["time"].shape == (8,)
    assert batch.patient_id == view.identifiers[:8], "strings come back as a list"
    assert batch.pad_mask == {}, "a table is never ragged"


# =========================================================================== #
# encoding and reporting
# =========================================================================== #


def test_feature_names_are_keyed_by_modality(cohort):
    """The shape the protocol declares, so a composite needs no special case.

    A flat list has nowhere to say which name belongs to which modality.
    """
    view = fitted_view(cohort)
    assert set(view.feature_names) == {"clinical"}
    assert view.feature_names == cohort.fit_preprocessor(cohort.identifiers).feature_names
    assert view.feature_names["clinical"][0] == CONTINUOUS[0]


def test_width_counts_every_modalitys_features(cohort):
    prep = cohort.fit_preprocessor(cohort.identifiers)
    assert prep.width == len(prep.feature_names["clinical"])
    assert prep.width == matrix_of(cohort.view(cohort.identifiers, prep)).shape[1]


def test_feature_names_belong_to_the_preprocessor_not_the_cohort(cohort):
    """The cohort declares columns; a fold produces features.

    A one-hot encoder fitted on one fold can emit a column the next one's does not, so
    "how many features" is only answerable once a fold exists.
    """
    view = fitted_view(cohort)
    assert cohort.feature_columns == CONTINUOUS + CATEGORICAL
    assert len(names(view)) > len(cohort.feature_columns), "one-hot expands the categoricals"
    assert len(names(view)) == matrix_of(view).shape[1]


def test_feature_names_are_readable(cohort):
    view = fitted_view(cohort)
    assert names(view)[: len(CONTINUOUS)] == CONTINUOUS
    assert "sex_male" in names(view)
    assert all("__" not in name for name in names(view)), "no transformer prefixes"


def test_describe_transforms_reports_what_the_cohort_declares(cohort):
    description = cohort.describe_transforms()
    assert "SimpleImputer -> StandardScaler" in description
    assert "SimpleImputer -> OneHotEncoder" in description


def test_the_preprocessor_describes_what_a_fold_applied(cohort):
    described = cohort.fit_preprocessor(ids_at(cohort, range(40))).describe()
    assert "fitted on 40 rows" in described
    assert "StandardScaler" in described


def test_a_passthrough_preprocessor_says_so(make_cohort):
    cohort = make_cohort(
        continuous=["biomarker"], continuous_transform=None, categorical=[], categorical_transform=None
    )
    assert "passthrough" in cohort.fit_preprocessor(ids_at(cohort, range(40))).describe()


def test_repr_surfaces_the_event_rate(cohort, frame):
    """So a mis-mapped ``event_value`` is visible the moment you print the cohort."""
    n_events = int(frame[EVENT_COLUMN].eq(EVENT_VALUE).sum())
    text = repr(cohort)

    assert "TabularCohort(" in text
    assert f"{len(cohort)} samples" in text
    assert f"{n_events} events" in text


def test_repr_without_a_target_omits_survival_summary(make_cohort):
    assert "events" not in repr(make_cohort(target=None))


# =========================================================================== #
# target integration
# =========================================================================== #


def test_the_target_is_bound_at_construction(cohort, frame):
    n_events = int(frame[EVENT_COLUMN].eq(EVENT_VALUE).sum())
    assert cohort.target.events_for(cohort.identifiers).sum() == n_events


def test_the_cohort_supplies_only_the_columns_the_target_declares(table_path, make_target):
    """``bind`` takes arrays, so the cohort must know which columns to hand over."""
    target = make_target()
    cohort = TabularCohort(table_path, identifier="patient_id", target=target)
    assert target.required_columns == (TIME_COLUMN, EVENT_COLUMN)
    assert len(cohort) == len(cohort.target.times_for(cohort.identifiers))


def test_a_mis_specified_event_value_fails_at_construction(table_path, make_target):
    """Not at training time, and not as a c-index of 0.31 three hours later."""
    with pytest.raises(ValueError, match="matches 0%"):
        TabularCohort(table_path, identifier="patient_id", target=make_target(event_value="dead"))


def test_a_target_over_a_missing_column_fails_at_construction(table_path):
    with pytest.raises(ValueError, match="requires column"):
        TabularCohort(
            table_path,
            identifier="patient_id",
            target=SurvivalTarget(time="no_such_column", event=EVENT_COLUMN, event_value=EVENT_VALUE),
        )


def test_columns_used_as_supervision_are_not_silently_also_features(cohort):
    """Declaring the outcome as a covariate would be a perfect, useless predictor."""
    view = fitted_view(cohort)
    assert TIME_COLUMN not in names(view)
    assert not any(name.startswith(EVENT_COLUMN) for name in names(view))


# --------------------------------------------------------------------------- #
# batch -- bulk access, the counterpart to iterating
# --------------------------------------------------------------------------- #


def test_batch_collates_every_sample_in_identifier_order(cohort):
    """The seam an embedder reads: the whole split as the object a DataLoader yields."""
    view = fitted_view(cohort)
    batch = view.batch()

    assert batch.patient_id == view.identifiers
    assert batch.modalities["clinical"].shape == (len(view), view.preprocessor.width)
    for i in range(len(view)):
        torch.testing.assert_close(batch.modalities["clinical"][i], view[i].modalities["clinical"])


def test_batch_carries_the_target_alongside_the_features(cohort):
    """One call gives an embedder both what to embed and what to condition on."""
    batch = fitted_view(cohort).batch()
    assert set(batch.target) == {"time", "event"}
    assert batch.target["event"].shape == (len(cohort),)


def test_batch_matches_a_full_size_dataloader_batch(cohort):
    """Bulk and streaming must not be two answers."""
    view = fitted_view(cohort)
    loaded = next(iter(DataLoader(view, batch_size=len(view), collate_fn=collate_samples)))
    torch.testing.assert_close(view.batch().modalities["clinical"], loaded.modalities["clinical"])


def test_batch_refuses_an_empty_view(cohort):
    with pytest.raises(ValueError, match="empty list of samples"):
        cohort.view([], cohort.fit_preprocessor(cohort.identifiers[:3])).batch()
