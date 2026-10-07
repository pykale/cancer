import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from kalecancer.loaddata import (
    BagModality,
    BaseTarget,
    Classification,
    FixedShapeModality,
    MultimodalDataset,
    PatchFeatures,
    Tabular,
    TimeToEvent,
    train_test_split,
)
from kalecancer.prepdata import ColumnGroup, TableTransform


def survival(cohort):
    return TimeToEvent(time=cohort.time, event=cohort.event)


def wsi(cohort, multiple_files="concatenate"):
    return PatchFeatures.from_glob(cohort.wsi_pattern, id_pattern=cohort.id_pattern, multiple_files=multiple_files)


def dataset(cohort, required=("clinical", "wsi"), target=None):
    return MultimodalDataset(
        modalities={"clinical": Tabular(cohort.clinical), "wsi": wsi(cohort)},
        target=survival(cohort) if target is None else target,
        required_modalities=list(required),
    )


# ---------------------------------------------------------------- identifiers


def test_integer_ids_raise_with_hint(cohort):
    clinical = cohort.clinical.reset_index(drop=True)
    with pytest.raises(TypeError, match="dtype=\\{'patient_id': str\\}"):
        Tabular(clinical)


def test_duplicate_ids_raise(cohort):
    clinical = pd.concat([cohort.clinical, cohort.clinical.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate ids"):
        Tabular(clinical)


def test_ids_differing_only_by_leading_zeros_raise(cohort):
    time = cohort.time.rename(index=lambda pid: pid.lstrip("0"))
    event = cohort.event.rename(index=lambda pid: pid.lstrip("0"))
    with pytest.raises(ValueError, match="leading zeros"):
        MultimodalDataset(
            {"clinical": Tabular(cohort.clinical)}, target=TimeToEvent(time, event), required_modalities=["clinical"]
        )


# ---------------------------------------------------------------- targets


def test_time_to_event_rejects_unmapped_status(cohort):
    status = pd.Series(np.where(cohort.event, "deceased", "living"), index=cohort.event.index)
    status.iloc[2] = "unknown"
    with pytest.raises(ValueError, match="not censoring"):
        TimeToEvent(cohort.time, status.map({"deceased": True, "living": False}))


def test_time_to_event_rejects_non_bool_event(cohort):
    with pytest.raises(TypeError, match="must be bool"):
        TimeToEvent(cohort.time, cohort.event.astype(int))


@pytest.mark.parametrize("bad", [0.0, -3.0, np.nan, np.inf])
def test_time_to_event_rejects_invalid_times(cohort, bad):
    time = cohort.time.copy()
    time.iloc[0] = bad
    with pytest.raises(ValueError, match="finite and > 0"):
        TimeToEvent(time, cohort.event)


def test_time_to_event_rejects_mismatched_ids(cohort):
    with pytest.raises(ValueError, match="same ids"):
        TimeToEvent(cohort.time, cohort.event.iloc[1:])


def test_time_to_event_tensors_follow_requested_order(cohort):
    target = survival(cohort)
    tensors = target.tensors(["003", "001"])
    assert tensors["time"].dtype == torch.float32 and tensors["event"].dtype == torch.bool
    assert tensors["time"][0].item() == pytest.approx(cohort.time["003"])
    assert tensors["event"].tolist() == [bool(cohort.event["003"]), bool(cohort.event["001"])]


def test_classification_validation(cohort):
    with pytest.raises(ValueError, match="not in classes"):
        Classification(cohort.labels, classes=["low", "medium"])
    labels = cohort.labels.copy()
    labels.iloc[0] = None
    with pytest.raises(ValueError, match="missing"):
        Classification(labels, classes=["low", "high"])
    with pytest.raises(ValueError, match="distinct"):
        Classification(cohort.labels, classes=["low", "low"])


def test_classification_indices_follow_classes_order(cohort):
    ids = ["001", "002", "003"]
    expected = [0 if cohort.labels[pid] == "low" else 1 for pid in ids]
    assert Classification(cohort.labels, classes=["low", "high"]).tensors(ids)["label"].tolist() == expected
    reversed_order = Classification(cohort.labels, classes=["high", "low"])
    assert reversed_order.tensors(ids)["label"].tolist() == [1 - k for k in expected]


def test_target_subclasses_must_implement_the_target_methods():
    class Incomplete(BaseTarget):
        def tensors(self, ids):
            return {}

    with pytest.raises(TypeError, match=r"abstract methods.*counts.*strata"):
        Incomplete()


# ---------------------------------------------------------------- patch features


def test_several_files_per_patient_raise_by_default(cohort):
    with pytest.raises(ValueError, match="multiple_files='concatenate'"):
        PatchFeatures.from_glob(cohort.wsi_pattern, id_pattern=cohort.id_pattern)


def test_concatenated_bag_and_coords_are_row_aligned_regardless_of_file_order(cohort):
    reversed_files = cohort.wsi_files.iloc[::-1]
    source = PatchFeatures(reversed_files, multiple_files="concatenate")
    bag = source["005"]
    coords = source.instances("005")
    np.testing.assert_array_equal(bag.numpy(), cohort.bags["005"])
    np.testing.assert_array_equal(coords[["x", "y"]].to_numpy(), cohort.coords["005"])
    assert coords["file"].nunique() == 2


def test_id_pattern_must_match_every_file(cohort):
    with pytest.raises(ValueError, match="does not match id_pattern"):
        PatchFeatures.from_glob(cohort.wsi_pattern, id_pattern=r"XX_(\d{3})\.h5$")


def test_missing_h5_key_raises(cohort):
    source = PatchFeatures(cohort.wsi_files, features_key="feats", multiple_files="concatenate")
    with pytest.raises(KeyError, match="no dataset 'feats'"):
        source["001"]


# ---------------------------------------------------------------- modalities


def test_one_id_gives_an_item_and_several_give_a_batch_in_the_order_asked(cohort):
    table = Tabular(cohort.clinical[["age"]])
    assert table["002"].shape == (1,)
    batch = table[["003", "001"]]
    assert batch.shape == (2, 1)
    assert batch[:, 0].tolist() == pytest.approx(cohort.clinical.loc[["003", "001"], "age"].tolist())

    bags = wsi(cohort)[["005", "001"]]
    assert isinstance(bags, list)
    assert [len(bag) for bag in bags] == [len(cohort.bags["005"]), len(cohort.bags["001"])]


def test_indexing_a_modality_checks_the_ids(cohort):
    table = Tabular(cohort.clinical[["age"]])
    with pytest.raises(ValueError, match="no ids requested"):
        table[[]]
    with pytest.raises(KeyError, match="unknown id '999'"):
        table["999"]
    with pytest.raises(KeyError, match="1 unknown ids"):
        table[["001", "999"]]


def test_datasets_take_modality_instances_only(cohort):
    with pytest.raises(TypeError, match=r"\['clinical'\] are not Modality instances\. Wrap a DataFrame in Tabular"):
        MultimodalDataset({"clinical": cohort.clinical}, target=None, required_modalities=[])


def test_modality_subclasses_must_implement_load_and_bags_their_instances():
    class NoLoad(FixedShapeModality):
        ids = pd.Index(["001"])

    class NoInstances(BagModality):
        ids = pd.Index(["001"])

        def _load(self, id):
            return torch.zeros(3, 4)

    with pytest.raises(TypeError, match="abstract method.*_load"):
        NoLoad()
    with pytest.raises(TypeError, match="abstract method.*instances"):
        NoInstances()


def test_a_new_fixed_shape_modality_needs_only_ids_and_load():
    class Constant(FixedShapeModality):
        def __init__(self, ids):
            self.ids = pd.Index(ids)

        def _load(self, id):
            return torch.full((2, 3), float(id))

    data = MultimodalDataset({"grid": Constant(["001", "002"])}, target=None, required_modalities=[])
    batch = data.collate([data[i] for i in range(len(data))])
    assert batch["inputs"]["grid"].shape == (2, 2, 3)
    assert batch["inputs"]["grid"][1, 0, 0].item() == 2.0


# ---------------------------------------------------------------- dataset


def test_required_modalities_define_inclusion_and_summary(cohort):
    data = dataset(cohort)
    assert len(data) == 30
    summary = data.summary()["count"]
    assert summary["included"] == 30
    assert summary["excluded: missing required modality 'wsi'"] == 10
    assert summary["events"] == int(cohort.event.loc[data.ids].sum())


def test_patients_without_a_modality_can_be_kept(cohort):
    data = dataset(cohort, required=["clinical"])
    assert len(data) == 40
    assert data.present["wsi"].sum() == 30
    assert not data.present.loc["040", "wsi"]


def test_patients_without_target_are_excluded(cohort):
    target = TimeToEvent(cohort.time.iloc[:20], cohort.event.iloc[:20])
    data = dataset(cohort, required=["clinical"], target=target)
    assert len(data) == 20
    assert data.summary()["count"]["excluded: no target"] == 20


def test_unknown_required_modality_raises(cohort):
    with pytest.raises(KeyError, match="unknown modalities"):
        MultimodalDataset({"clinical": Tabular(cohort.clinical)}, target=None, required_modalities=["wsi"])


def test_subset_raises_on_excluded_ids_with_reason(cohort):
    data = dataset(cohort)
    with pytest.raises(KeyError, match="missing required modality 'wsi'"):
        data.subset(["001", "040"])
    with pytest.raises(KeyError, match="not in any modality or target"):
        data.subset(["999"])
    with pytest.raises(TypeError, match="must be str"):
        data.subset([1, 2])


def test_subset_is_sorted_and_recorded(cohort):
    part = dataset(cohort).subset(["010", "002", "005"])
    assert part.ids == ["002", "005", "010"]
    assert part.summary()["count"]["excluded: outside subset"] == 27


def test_batch_contract(cohort):
    data = dataset(cohort, required=["clinical"]).subset(["001", "005", "039", "040"])
    clinical = cohort.clinical.assign(sex=0.0, smoking=0.0)
    data = MultimodalDataset(
        {"clinical": Tabular(clinical), "wsi": wsi(cohort)}, target=survival(cohort), required_modalities=["clinical"]
    ).subset(data.ids)
    batch = data.collate([data[i] for i in range(len(data))])
    assert batch["ids"] == ["001", "005", "039", "040"]
    assert batch["present"]["wsi"].tolist() == [True, True, False, False]
    assert batch["present"]["clinical"].tolist() == [True] * 4
    assert batch["inputs"]["clinical"].shape == (4, 3)
    assert [bag.shape[0] for bag in batch["inputs"]["wsi"]] == [len(cohort.bags["001"]), len(cohort.bags["005"])]
    assert batch["target"]["event"].dtype == torch.bool and batch["target"]["time"].shape == (4,)


def test_a_modality_no_patient_in_the_batch_has_gets_no_input(cohort):
    numeric = cohort.clinical[["age"]]
    data = MultimodalDataset(
        {"clinical": Tabular(numeric), "wsi": wsi(cohort)}, target=None, required_modalities=["clinical"]
    ).subset(["039", "040"])
    batch = data.collate([data[i] for i in range(len(data))])
    assert batch["present"]["wsi"].tolist() == [False, False]
    assert list(batch["inputs"]) == ["clinical"]


def test_train_test_split_is_stratified_and_disjoint(cohort):
    data = dataset(cohort, required=["clinical"])
    train, test = train_test_split(data, test_size=0.25, stratify=True, random_state=0)
    assert set(train.ids).isdisjoint(test.ids) and len(train) + len(test) == 40
    rate = cohort.event.mean()
    assert abs(cohort.event.loc[test.ids].mean() - rate) < 0.15


# ---------------------------------------------------------------- tables and transforms


def test_non_numeric_table_needs_a_transform(cohort):
    data = MultimodalDataset({"clinical": Tabular(cohort.clinical)}, target=None, required_modalities=[])
    with pytest.raises(TypeError, match="give this modality a transform"):
        data[0]


def test_nan_in_table_loads_as_nan(cohort):
    # the table only reports its values: the stage list that reads them decides whether NaN is acceptable
    numeric = pd.DataFrame({"age": cohort.clinical["age"]})
    numeric.iloc[1, 0] = np.nan
    data = MultimodalDataset({"clinical": Tabular(numeric)}, target=None, required_modalities=[])
    assert torch.isnan(data[1]["inputs"]["clinical"]).all()


def column_transform():
    return TableTransform(
        groups={
            "numeric": ColumnGroup(["age"], [StandardScaler()]),
            "categorical": ColumnGroup(
                ["sex", "smoking"], [SimpleImputer(strategy="constant", fill_value="missing"), OrdinalEncoder()]
            ),
        },
        drop=[],
    )


def test_table_transform_output_feeds_the_dataset(cohort):
    data = MultimodalDataset({"clinical": Tabular(cohort.clinical)}, target=None, required_modalities=[])
    fitted = clone(column_transform()).fit(data.modalities["clinical"].transform_input(data.ids))
    transformed = data.with_transforms({"clinical": fitted})
    assert transformed[0]["inputs"]["clinical"].shape == (3,)
    assert transformed[0]["inputs"]["clinical"].dtype == torch.float32
    assert list(transformed.modalities["clinical"].frame.columns) == ["age", "sex", "smoking"]


def test_table_transform_requires_every_column(cohort):
    transform = TableTransform(groups={"numeric": ColumnGroup(["age"], [StandardScaler()])}, drop=["sex"])
    with pytest.raises(ValueError, match="unlisted=\\['smoking'\\]"):
        transform.fit(cohort.clinical)


def test_table_transform_clone_is_unfitted_and_independent(cohort):
    transform = column_transform().fit(cohort.clinical)
    copy = clone(transform)
    assert not hasattr(copy, "transformer_")
    assert copy.groups["numeric"].steps[0] is not transform.groups["numeric"].steps[0]


@pytest.mark.parametrize(
    ("transform", "columns"),
    [(StandardScaler(), ["age", "size"]), (PCA(n_components=1), ["pca0"]), (FunctionTransformer(), [0, 1])],
)
def test_transformed_tables_keep_the_transform_column_names(transform, columns):
    # FunctionTransformer has no feature names, so the columns fall back to positions
    frame = pd.DataFrame({"age": [50.0, 60.0, 70.0], "size": [1.0, 2.0, 4.0]}, index=["001", "002", "003"])
    table = Tabular(frame)
    fitted = clone(transform).fit(table.transform_input(table.ids))
    assert list(table.with_transform(fitted, table.ids).frame.columns) == columns


def test_bag_modalities_do_not_accept_transforms(cohort):
    data = dataset(cohort)
    with pytest.raises(TypeError, match="PatchFeatures"):
        data.with_transforms({"wsi": StandardScaler()})
    with pytest.raises(TypeError, match="PatchFeatures does not accept transforms"):
        data.modalities["wsi"].transform_input(data.ids)
