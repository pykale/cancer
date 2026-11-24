import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from kale.predict.decode import VCDN
from kale.prepdata.tabular_transform import ToOneHotEncoding, ToTensor
from torch.nn import CrossEntropyLoss

from kalecancer.interpret.model_weights import select_top_features_by_masking
from kalecancer.loaddata.multiomics_datasets import SparseMultiomicsDataset
from kalecancer.model.encode.mogonet import MogonetGCN
from kalecancer.model.heads import LinearClassifier
from kalecancer.pipeline.multiomics_trainer import MultiomicsTrainer

NUM_MODALITIES = 3
NUM_FEATS = 4
NUM_TRAIN, NUM_TEST = 18, 9


def _write_omics(raw_dir, num_classes, rng):
    """Write a pre-split omics dataset in the five-files-per-modality layout the dataset reads."""
    raw_dir.mkdir(parents=True)
    labels = {split: np.arange(n) % num_classes for split, n in (("tr", NUM_TRAIN), ("te", NUM_TEST))}
    file_names = []
    for modality in range(1, NUM_MODALITIES + 1):
        for split in ("tr", "te"):
            # shift each class so the features carry signal
            data = rng.normal(size=(len(labels[split]), NUM_FEATS)) + labels[split][:, None]
            np.savetxt(raw_dir / f"{modality}_{split}.csv", data, delimiter=",")
            np.savetxt(raw_dir / f"{modality}_lbl_{split}.csv", labels[split], fmt="%d")
        pd.Series([f"m{modality}_f{i}" for i in range(NUM_FEATS)]).to_csv(
            raw_dir / f"{modality}_feat_name.csv", header=False, index=False
        )
        file_names += [f"{modality}_{s}.csv" for s in ("tr", "lbl_tr", "te", "lbl_te", "feat_name")]
    return file_names


@pytest.fixture
def mogonet(tmp_path, num_classes):
    rng = np.random.default_rng(0)
    file_names = _write_omics(tmp_path / "raw", num_classes, rng)
    dataset = SparseMultiomicsDataset(
        root=str(tmp_path),
        raw_file_names=file_names,
        num_modalities=NUM_MODALITIES,
        num_classes=num_classes,
        edge_per_node=2,
        url=None,
        random_split=False,
        equal_weight=False,
        pre_transform=ToTensor(dtype=torch.float),
        target_pre_transform=ToOneHotEncoding(dtype=torch.float),
    )
    torch.manual_seed(2023)
    hidden = [8, 8, 4]
    model = MultiomicsTrainer(
        dataset=dataset,
        num_modalities=NUM_MODALITIES,
        num_classes=num_classes,
        unimodal_encoder=[MogonetGCN(NUM_FEATS, hidden, dropout=0.5) for _ in range(NUM_MODALITIES)],
        unimodal_decoder=[LinearClassifier(hidden[-1], num_classes) for _ in range(NUM_MODALITIES)],
        loss_fn=CrossEntropyLoss(reduction="none"),
        multimodal_decoder=VCDN(NUM_MODALITIES, num_classes, hidden_dim=pow(num_classes, NUM_MODALITIES)),
    )
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    return trainer, model, dataset


@pytest.mark.parametrize("num_classes, metric", [(2, "F1"), (3, "F1 macro")])
def test_select_top_features_by_masking(mogonet, num_classes, metric):
    trainer, model, dataset = mogonet
    before = [dataset.get(m).x.clone() for m in range(NUM_MODALITIES)]

    top = select_top_features_by_masking(trainer, model, dataset, metric=metric, num_top_feats=5)

    assert list(top.columns) == ["feat_name", "imp", "omics"]
    assert len(top) == 5
    assert top["imp"].is_monotonic_decreasing
    assert set(top["omics"]) <= set(range(NUM_MODALITIES))
    assert all(name.startswith(f"m{omics + 1}_") for name, omics in zip(top["feat_name"], top["omics"], strict=False))
    # every masked feature is restored once its modality is done
    for modality in range(NUM_MODALITIES):
        assert torch.equal(dataset.get(modality).x, before[modality])


@pytest.mark.parametrize("num_classes", [2])
def test_select_top_features_by_masking_rejects_unknown_metric(mogonet):
    trainer, model, dataset = mogonet
    with pytest.raises(ValueError, match="nonexistent_metric"):
        select_top_features_by_masking(trainer, model, dataset, metric="nonexistent_metric")
