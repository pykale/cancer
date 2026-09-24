import json
from functools import partial

import pandas as pd
import pytest
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn

from kalecancer.evaluate import HarrellC, UnoC
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, TimeToEvent
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion
from kalecancer.pipeline import EarlyStopping, Pipeline, dump_config, load_pipeline
from kalecancer.prepdata import ColumnGroup, TableTransform


def build_pipeline(metric=None):
    model = IntermediateFusion(
        encoding={
            "clinical": [nn.Linear(3, 4)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 4)],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=8, ties="efron"),
    )
    transform = TableTransform(
        groups={
            "numeric": ColumnGroup(["age"], [StandardScaler()]),
            "categorical": ColumnGroup(
                ["sex", "smoking"],
                [
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ],
            ),
        },
        drop=[],
    )
    return Pipeline(
        model=model,
        transforms={"clinical": transform},
        optimizer=partial(torch.optim.AdamW, lr=1e-2, weight_decay=0.01),
        batch_size=8,
        drop_last=True,
        max_epochs=2,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0),
        early_stopping=EarlyStopping(metric=metric or HarrellC(), patience=2, restore_best=True),
        accelerator="cpu",
        precision="32-true",
        random_state=0,
        param_groups={"encoding.wsi": {"lr": 1e-3}},
    )


def same(a, b):
    # key order is part of the config: it fixes feature and concatenation order
    return json.dumps(a) == json.dumps(b)


def test_dump_load_dump_is_a_fixed_point():
    first = dump_config(build_pipeline(metric=UnoC(tau=500.0)))
    assert first["class_path"] == "kalecancer.pipeline.Pipeline"
    assert first["init_args"]["model"]["init_args"]["head"]["init_args"] == {"in_dim": 8, "ties": "efron"}
    assert first["init_args"]["optimizer"]["init_args"]["lr"] == 1e-2
    second = dump_config(load_pipeline(first))
    assert same(first, second)


def test_pipeline_from_yaml_trains_like_the_python_pipeline(cohort, tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(dump_config(build_pipeline()), sort_keys=False))
    loaded = load_pipeline(path)

    data = MultimodalDataset(
        {"clinical": cohort.clinical, "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate")},
        target=TimeToEvent(cohort.time, cohort.event),
        required_modalities=["clinical", "wsi"],
    )
    train, test = data.subset(data.ids[:22]), data.subset(data.ids[22:])
    pd.testing.assert_frame_equal(build_pipeline().fit(train).predict(test), loaded.fit(train).predict(test))


def config_with(change):
    config = dump_config(build_pipeline())
    change(config["init_args"])
    return config


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(batchsize=8),
        lambda c: c["model"]["init_args"]["head"]["init_args"].update(ties="efronn"),
        lambda c: c["transforms"]["clinical"]["init_args"]["groups"]["numeric"]["init_args"]["steps"][0].update(
            init_args={"with_meen": False}
        ),
        lambda c: c["model"]["init_args"]["encoding"]["wsi"][0].update(class_path="kalecancer.ABMILL"),
        lambda c: c.pop("random_state"),
    ],
    ids=["unknown argument", "invalid literal", "nested sklearn typo", "unknown class", "missing argument"],
)
def test_invalid_configs_raise(change):
    with pytest.raises(Exception):  # noqa: B017 - jsonargparse reports these as ArgumentError or TypeError
        load_pipeline(config_with(change))


def test_components_must_store_their_arguments():
    class Scale(nn.Module):
        def __init__(self, factor: float):
            super().__init__()
            self.weight = nn.Parameter(torch.full((3,), factor))

    pipe = build_pipeline()
    pipe.model = IntermediateFusion(
        encoding={"clinical": [Scale(2.0), nn.Linear(3, 4)], "wsi": [ABMIL(16, 4, 4, 0.0)]},
        fusion=Concat(),
        head=CoxHead(in_dim=8, ties="efron"),
    )
    with pytest.raises(TypeError, match="does not store its argument 'factor'"):
        dump_config(pipe)
