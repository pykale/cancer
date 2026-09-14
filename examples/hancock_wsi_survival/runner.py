"""End-to-end orchestration of this experiment.

Takes a configuration and runs the full workflow: cohort matching, leakage-safe
splitting, training, evaluation, and attention export.

This lives in the example rather than in ``kalecancer`` because it is an experiment,
not a library component: it names a survival endpoint, an attention export and a set
of configuration keys, none of which generalise to the next study. The library
supplies the pieces it composes -- cohort building, splitting,
:class:`~kalecancer.pipeline.CohortTrainer`, the metrics -- and none of them know
this workflow exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from examples.hancock.clinical import EndpointSpec
from examples.hancock.cohort import build_cohort, cohort_summary, log_cohort_summary, split_summary
from examples.hancock.dataset import resolve_paths
from kalecancer.evaluate.survival_predictions import (
    evaluate_predictions,
    predict_split,
    save_predictions,
    summarise_folds,
)
from kalecancer.interpret.attention import export_attention
from kalecancer.loaddata import ColumnTarget, FeatureBagSource, MultimodalDataset
from kalecancer.loaddata.multimodal_access import collate_ragged, release_workers
from kalecancer.loaddata.splitting import CohortSplitter, CrossValidation, HoldOut, Predefined
from kalecancer.model.embed import BagEncoder
from kalecancer.model.layers import AttentionMIL
from kalecancer.pipeline import CohortTrainer, SurvivalTask
from kalecancer.utils.io import ensure_dir, write_json
from kalecancer.utils.seed import seed_worker, set_seed

logger = logging.getLogger(__name__)

#: The single modality this experiment carries. Named so the embedder, the batch and
#: the attention export agree on it without any of them hardcoding a string.
MODALITY = "wsi"


def build_encoder(cfg: CfgNode) -> BagEncoder:
    """Attention MIL over patch features, wrapped for the trainer's embedder slot.

    :class:`~kalecancer.model.embed.BagEncoder` is what lets a bag be an ordinary
    modality: it pools a patient's patches into one vector and keeps the attention
    weights, so interpretation can read them back afterwards.
    """
    return BagEncoder(
        AttentionMIL(
            input_dim=cfg.MODEL.INPUT_DIM,
            hidden_dim=cfg.MODEL.HIDDEN_DIM,
            attention_dim=cfg.MODEL.ATTENTION_DIM,
            dropout=cfg.MODEL.DROPOUT,
            gated=cfg.MODEL.GATED,
        )
    )


class PipelineError(RuntimeError):
    """Raised when a pipeline cannot run with the given configuration."""


def _build_loader(dataset: MultimodalDataset, cfg: CfgNode, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_ragged,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
    )


def _build_datasets(cohort: pd.DataFrame, split: dict, cfg: CfgNode, group_key: str) -> dict[str, MultimodalDataset]:
    """Training subsamples patches to bound memory; evaluation keeps whole bags.

    ``train_eval`` re-reads the training patients without subsampling, so the baseline
    hazard and censoring weights come from the same full-bag predictions the evaluated
    splits produce.
    """

    def dataset(name: str, max_patches: int | None = None) -> MultimodalDataset:
        rows = cohort.iloc[split[name]]
        # The slide-level table collapses to one bag per patient; a second region
        # would simply be a second source here.
        paths = {patient: list(group["path"]) for patient, group in rows.groupby(group_key)}
        source = FeatureBagSource(
            paths,
            feature_dim=cfg.MODEL.INPUT_DIM,
            max_patches=max_patches,
            seed=cfg.SOLVER.SEED,
            with_coordinates=True,
        )
        target = ColumnTarget(
            rows.drop_duplicates(group_key),
            columns={"time": "duration", "event": "event"},
            id_column=group_key,
        )
        return MultimodalDataset(sorted(paths), {MODALITY: source}, target=target)

    return {
        "train": dataset("train", max_patches=cfg.DATASET.MAX_PATCHES or None),
        "train_eval": dataset("train"),
        "val": dataset("val"),
        "test": dataset("test"),
    }


def build_splitter(
    cfg: CfgNode,
    group_key: str = "patient_id",
    assignment: Mapping[str, Sequence[str]] | None = None,
) -> CohortSplitter:
    """The splitter ``DATASET.SPLIT_MODE`` describes.

    ``published`` is the default because a published assignment is what makes a
    number comparable with other work on the same cohort; a fresh random split
    silently gives up that comparability. Validation always comes out of the training
    half, so the test set is untouched whichever mode is chosen.

    The cohort is slide-level, so ``group_by`` is set throughout: a patient's slides
    must not span the train/validation carve any more than they may span train and
    test.

    Args:
        cfg: Pipeline configuration.
        group_key: Column whose samples must not span two splits.
        assignment: The dataset's published assignment, needed by ``published``.

    Raises:
        PipelineError: If ``SPLIT_MODE`` is unknown, or is ``published`` with no
            assignment to apply.
    """
    options = {
        "group_by": group_key,
        "stratify_by": list(cfg.DATASET.STRATIFY_KEYS),
        "val_size": cfg.DATASET.VAL_RATIO,
        "random_state": cfg.SOLVER.SEED,
    }
    mode = cfg.DATASET.SPLIT_MODE

    if mode == "published":
        if assignment is None:
            raise PipelineError(
                "DATASET.SPLIT_MODE is 'published' but this dataset supplied no assignment; "
                "pass one, or set DATASET.SPLIT_MODE to 'random' to draw a fresh split"
            )
        return Predefined(assignment, id_column=group_key, **options)
    if mode == "cv":
        return CrossValidation(n_splits=cfg.DATASET.NUM_FOLDS, **options)
    if mode == "random":
        return HoldOut(test_size=cfg.DATASET.TEST_RATIO, **options)
    raise PipelineError(f"unknown DATASET.SPLIT_MODE {mode!r}; expected 'published', 'cv' or 'random'")


def run_split(
    cohort: pd.DataFrame, split: dict, cfg: CfgNode, out_dir: str | Path, group_key: str = "patient_id"
) -> dict:
    """Train and evaluate one train/validation/test split.

    Args:
        cohort: Cohort table the split indices refer to.
        split: Positional indices keyed by split name.
        cfg: Pipeline configuration.
        out_dir: Directory for this run's artefacts.

    Returns:
        Metrics keyed by split name.
    """
    out_dir = ensure_dir(out_dir)
    logger.info("split sizes: %s", split_summary(cohort, split, group_key))
    datasets = _build_datasets(cohort, split, cfg, group_key)

    train_loader = _build_loader(datasets["train"], cfg, shuffle=True)
    val_loader = _build_loader(datasets["val"], cfg, shuffle=False)
    test_loader = _build_loader(datasets["test"], cfg, shuffle=False)

    model = CohortTrainer(
        {MODALITY: build_encoder(cfg)},
        task=SurvivalTask(),
        fusion_dim=cfg.MODEL.HIDDEN_DIM,
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
    )

    checkpoint = ModelCheckpoint(
        dirpath=Path(out_dir) / "checkpoints",
        filename="best",
        monitor="valid_c_index",
        mode="max",
        save_top_k=1,
    )
    callbacks: list[pl.Callback] = [checkpoint]
    if cfg.SOLVER.EARLY_STOP:
        callbacks.append(EarlyStopping(monitor="valid_c_index", mode="max", patience=cfg.SOLVER.EARLY_STOP))

    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        devices=cfg.SOLVER.DEVICES,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(out_dir), name="history"),
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)
    if checkpoint.best_model_path:
        # Weights only: the embedders are modules built here, not hyperparameters a
        # checkpoint could reconstruct.
        best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["state_dict"])

    train_eval_loader = _build_loader(datasets["train_eval"], cfg, shuffle=False)
    predictions = [
        predict_split(model, train_eval_loader, "train"),
        predict_split(model, val_loader, "val"),
        predict_split(model, test_loader, "test"),
    ]
    eval_times = list(cfg.SURVIVAL.EVAL_TIMES)
    metrics = {
        prediction.split: evaluate_predictions(prediction, predictions[0], eval_times) for prediction in predictions
    }

    save_predictions(out_dir, predictions, metrics)
    export_attention(model, test_loader, Path(out_dir) / "attention", modality=MODALITY, top_k=cfg.OUTPUT.TOP_K)

    # Every loader this split built is finished with. Released here rather than left
    # to the garbage collector, which would otherwise finalise a fold's worker pools
    # during interpreter shutdown; see release_workers for why that misbehaves.
    release_workers(train_loader, val_loader, test_loader, train_eval_loader)

    logger.info("test metrics: %s", metrics["test"])
    return metrics


def run(
    cfg: CfgNode,
    endpoint: EndpointSpec,
    fetch: Callable[[], tuple[Path, Path]] | None = None,
    splits: Callable[[], Mapping[str, Sequence[str]]] | None = None,
) -> dict:
    """Run the WSI survival pipeline described by ``cfg``.

    Args:
        cfg: Pipeline configuration, see :func:`kalecancer.config.get_cfg_defaults`.
        endpoint: How the clinical columns define the survival endpoint.
        fetch: Supplies the data when ``DATASET.SOURCE`` is not ``"local"``.
        splits: Supplies the dataset's published train/test assignment, which
            ``DATASET.SPLIT_MODE="published"`` -- the default -- needs. Deferred
            behind a callable for the same reason as ``fetch``: reading it may mean
            a download, and the cross-validation mode never needs it.

    Returns:
        Metrics for the run; cross-validation additionally reports fold aggregates.

    Raises:
        PipelineError: If no patients match the configured inputs, or the split
            cannot be built as configured.
    """
    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    group_key = cfg.DATASET.GROUP_KEY
    feature_root, clinical_path = resolve_paths(cfg, fetch)
    cohort = build_cohort(
        feature_root=feature_root,
        clinical_path=clinical_path,
        endpoint=endpoint,
        expected_dim=cfg.MODEL.INPUT_DIM,
        validate_features=cfg.DATASET.VALIDATE_FEATURES,
    )
    summary = cohort_summary(cohort, group_key)
    log_cohort_summary(summary)
    write_json(out_dir / "cohort_summary.json", summary)

    if cohort.empty:
        raise PipelineError(
            f"no patients matched between {feature_root} and {clinical_path}; see {out_dir / 'cohort_summary.json'}"
        )

    # Only read when the mode needs it, so a cross-validated run costs no download.
    assignment = splits() if splits is not None and cfg.DATASET.SPLIT_MODE == "published" else None

    splits_ = list(build_splitter(cfg, group_key, assignment).split(cohort))

    if len(splits_) == 1:
        metrics = run_split(cohort, splits_[0], cfg, out_dir, group_key)
    else:
        fold_metrics = [
            run_split(cohort, split, cfg, out_dir / f"fold_{index}", group_key) for index, split in enumerate(splits_)
        ]
        cross_validated = summarise_folds(fold_metrics)
        metrics = {"cross_validated": cross_validated, "folds": fold_metrics}
        write_json(out_dir / "metrics.json", metrics)
        # A fold whose test split holds no comparable pair reports no C-index, so the
        # headline is only logged when at least one fold produced one.
        c_index = cross_validated.get("c_index")
        if c_index:
            logger.info("cross-validated test C-index: %.4f +/- %.4f", c_index["mean"], c_index["std"])
        else:
            logger.warning("no fold produced a C-index; see %s", out_dir / "metrics.json")

    logger.info("artefacts written to %s", out_dir)
    return metrics
