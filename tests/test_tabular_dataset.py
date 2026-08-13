"""Tests for ``TabularDataset``: column-role validation, loading, and row access."""

from __future__ import annotations

import pandas as pd
import pytest

from kalecancer.loaddata.tabular_dataset import TabularDataset

# "zeta" sits before "alpha" on purpose: any test that asserts file order would still pass if the
# columns were sorted, unless the file itself is out of alphabetical order.
FRAME = pd.DataFrame(
    {
        "pid": ["p1", "p2", "p3"],
        "zeta": [10, 20, 30],
        "alpha": [1.5, 2.5, 3.5],
        "y": [0, 1, 0],
    }
)


@pytest.fixture
def write_table(tmp_path):
    """Write a frame to a temp file, picking the format from the file name."""

    def _write(frame: pd.DataFrame = FRAME, name: str = "cohort.csv"):
        path = tmp_path / name
        if path.suffix.lower() == ".json":
            frame.to_json(path, orient="records")
        else:
            frame.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def table(write_table):
    return write_table()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"identifier_column": []}, "at least one column"),
        ({"identifier_column": None}, "at least one column"),
        ({"identifier_column": "pid", "target_column": "y", "feature_columns": ["alpha", "y"]}, "feature and target"),
        ({"identifier_column": "pid", "feature_columns": ["pid", "alpha"]}, "become the index"),
        ({"identifier_column": "pid", "target_column": "pid"}, "become the index"),
        ({"identifier_column": "pid", "feature_columns": object()}, "sequence of labels"),
        ({"identifier_column": "pid", "feature_columns": [1.5]}, "only str or int labels"),
    ],
    ids=["empty_ids", "no_ids", "target_as_feature", "id_as_feature", "id_as_target", "not_a_sequence", "bad_label"],
)
def test_incoherent_column_roles_are_rejected(table, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TabularDataset(table, **kwargs)


def test_column_roles_are_checked_before_the_file_is_opened(tmp_path):
    with pytest.raises(ValueError, match="feature and target"):
        TabularDataset(tmp_path / "does_not_exist.csv", "pid", target_column="y", feature_columns=["y"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"identifier_column": "ghost"}, "no identifier column"),
        ({"identifier_column": "pid", "target_column": "ghost"}, "no target column"),
        ({"identifier_column": "pid", "feature_columns": "ghost"}, "no feature column"),
    ],
    ids=["identifier", "target", "feature"],
)
def test_columns_missing_from_the_file_are_reported_by_role(table, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TabularDataset(table, **kwargs).load_data()


def test_a_missing_column_error_names_what_the_file_does_hold(table):
    with pytest.raises(ValueError, match=r"Available columns: \['zeta', 'alpha', 'y'\]"):
        TabularDataset(table, "pid", target_column="ghost").load_data()


def test_duplicate_identifiers_are_rejected(write_table):
    frame = pd.DataFrame({"pid": ["p1", "p1"], "v": [1, 2]})

    with pytest.raises(ValueError, match="one row per identifier"):
        TabularDataset(write_table(frame), "pid").load_data()


def test_unsupported_file_types_are_rejected(tmp_path):
    path = tmp_path / "cohort.parquet"
    path.touch()

    with pytest.raises(ValueError, match="Unsupported file type"):
        TabularDataset(path, "pid").load_data()


@pytest.mark.parametrize("name", ["cohort.csv", "cohort.json", "cohort.CSV", "cohort.JSON"])
def test_both_formats_load_whatever_the_suffix_case(write_table, name):
    dataset = TabularDataset(write_table(name=name), "pid", target_column="y")
    dataset.load_data()

    assert dataset.identifiers == ["p1", "p2", "p3"]
    assert dataset.feature_columns == ["zeta", "alpha"]


def test_inferred_features_follow_file_order_not_alphabetical(table):
    dataset = TabularDataset(table, "pid", target_column="y")
    dataset.load_data()

    assert dataset.feature_columns == ["zeta", "alpha"]


def test_explicit_features_keep_the_order_they_were_asked_for(table):
    dataset = TabularDataset(table, "pid", target_column="y", feature_columns=["alpha", "zeta"])

    features, _ = dataset.get_by_id("p1")
    assert list(features.index) == ["alpha", "zeta"]


@pytest.mark.parametrize("sequence", [list, tuple], ids=["list", "tuple"])
def test_any_sequence_of_labels_is_accepted(table, sequence):
    dataset = TabularDataset(
        table,
        sequence(["pid"]),
        target_column=sequence(["y"]),
        feature_columns=sequence(["zeta"]),
    )

    features, targets = dataset.get_by_id("p1")
    assert list(features.index) == ["zeta"]
    assert list(targets.index) == ["y"]


def test_a_string_label_is_not_split_into_characters(table):
    dataset = TabularDataset(table, "pid")
    dataset.load_data()

    assert dataset.identifier_column == ["pid"]


def test_get_by_id_loads_on_demand(table):
    features, targets = TabularDataset(table, "pid", target_column="y").get_by_id("p2")

    assert features["zeta"] == 20
    assert targets["y"] == 1


def test_without_a_target_only_features_come_back(table):
    features = TabularDataset(table, "pid").get_by_id("p1")

    assert isinstance(features, pd.Series)
    assert list(features.index) == ["zeta", "alpha", "y"]


def test_with_a_target_features_and_targets_come_back_apart(table):
    features, targets = TabularDataset(table, "pid", target_column="y").get_by_id("p3")

    assert isinstance(features, pd.Series) and isinstance(targets, pd.Series)
    assert "y" not in features.index
    assert targets.to_dict() == {"y": 0}


def test_several_targets_come_back_together(table):
    dataset = TabularDataset(table, "pid", target_column=["y", "alpha"], feature_columns=["zeta"])

    _, targets = dataset.get_by_id("p2")
    assert list(targets.index) == ["y", "alpha"]


def test_columns_with_no_role_are_left_out(write_table):
    dataset = TabularDataset(
        write_table(FRAME.assign(scratch=["a", "b", "c"])),
        "pid",
        target_column="y",
        feature_columns=["zeta"],
    )

    features, _ = dataset.get_by_id("p1")
    assert list(features.index) == ["zeta"]


def test_columns_with_no_role_are_not_held_in_memory(table):
    # Reaching into _table on purpose. Dropping unused columns keeps a wide clinical file from
    # being retained in full, which matters here but has no public surface to assert against.
    dataset = TabularDataset(table, "pid", target_column="y", feature_columns=["zeta"])
    dataset.load_data()

    assert list(dataset._table.columns) == ["zeta", "y"]


def test_getitem_walks_the_same_rows_by_position(table):
    dataset = TabularDataset(table, "pid", target_column="y")

    assert len(dataset) == 3
    positional_features, positional_targets = dataset[1]
    expected_features, expected_targets = dataset.get_by_id("p2")
    assert positional_features.equals(expected_features)
    assert positional_targets.equals(expected_targets)


def test_reloading_re_infers_features_from_the_current_file(write_table):
    path = write_table()
    dataset = TabularDataset(path, "pid", target_column="y")
    dataset.load_data()
    assert dataset.feature_columns == ["zeta", "alpha"]

    # The file gains a column. _loader is called directly to get past the once-only guard in
    # load_data; it has to re-infer from the file rather than reuse the list it inferred last time,
    # which is why the caller's request is kept in a separate attribute.
    FRAME.assign(extra=[1, 2, 3]).to_csv(path, index=False)
    dataset._loader()

    assert dataset.feature_columns == ["zeta", "alpha", "extra"]


def test_several_identifier_columns_form_a_composite_key(write_table):
    # "a" repeats in the first column, so this also pins that duplicates are judged on the
    # combination of identifier columns rather than on any one of them.
    frame = pd.DataFrame({"site": ["a", "a", "b"], "n": [1, 2, 1], "v": [7, 8, 9]})
    dataset = TabularDataset(write_table(frame), ["site", "n"], target_column="v")
    dataset.load_data()

    assert dataset.identifiers == [("a", 1), ("a", 2), ("b", 1)]
    _, targets = dataset.get_by_id(("a", 2))
    assert targets["v"] == 8
