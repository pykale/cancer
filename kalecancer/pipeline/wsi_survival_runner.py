"""End-to-end orchestration of the WSI survival pipeline.

Takes a configuration and runs the full workflow: cohort matching, leakage-safe
splitting, training, evaluation, and attention export.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from kalecancer.evaluate.survival_report import (
    evaluate_predictions,
    predict_split,
    save_survival_report,
    summarise_folds,
)
from kalecancer.interpret.attention import export_attention
from kalecancer.loaddata.cohort import build_cohort
from kalecancer.loaddata.hancock import resolve_dataset
from kalecancer.loaddata.split import CohortSplit, split_patients, stratified_patient_folds
from kalecancer.loaddata.wsi_dataset import WSIFeatureBagDataset, collate_bags
from kalecancer.pipeline.wsi_survival_trainer import WSISurvivalTrainer
from kalecancer.utils.io import ensure_dir, write_json
from kalecancer.utils.seed import seed_worker, set_seed

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when a pipeline cannot run with the given configuration."""


def _build_loader(dataset: WSIFeatureBagDataset, cfg: CfgNode, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_bags,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
    )


def _build_datasets(split: CohortSplit, cfg: CfgNode) -> dict[str, WSIFeatureBagDataset]:
    """Training subsamples patches to bound memory; evaluation keeps whole bags.

    ``train_eval`` re-reads the training patients without subsampling, so the baseline
    hazard and censoring weights come from the same full-bag predictions the evaluated
    splits produce.
    """
    max_patches = cfg.DATASET.MAX_PATCHES or None
    return {
        "train": WSIFeatureBagDataset(
            split.train, expected_dim=cfg.MODEL.INPUT_DIM, max_patches=max_patches, seed=cfg.SOLVER.SEED
        ),
        "train_eval": WSIFeatureBagDataset(split.train, expected_dim=cfg.MODEL.INPUT_DIM),
        "val": WSIFeatureBagDataset(split.val, expected_dim=cfg.MODEL.INPUT_DIM),
        "test": WSIFeatureBagDataset(split.test, expected_dim=cfg.MODEL.INPUT_DIM),
    }


def run_split(split: CohortSplit, cfg: CfgNode, out_dir: str | Path) -> dict:
    """Train and evaluate one train/validation/test split.

    Args:
        split: Disjoint patient-level splits.
        cfg: Pipeline configuration.
        out_dir: Directory for this run's artefacts.

    Returns:
        Metrics keyed by split name.
    """
    out_dir = ensure_dir(out_dir)
    logger.info("split sizes: %s", split.sizes())
    datasets = _build_datasets(split, cfg)

    train_loader = _build_loader(datasets["train"], cfg, shuffle=True)
    val_loader = _build_loader(datasets["val"], cfg, shuffle=False)
    test_loader = _build_loader(datasets["test"], cfg, shuffle=False)

    model = WSISurvivalTrainer(
        input_dim=cfg.MODEL.INPUT_DIM,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        attention_dim=cfg.MODEL.ATTENTION_DIM,
        dropout=cfg.MODEL.DROPOUT,
        gated=cfg.MODEL.GATED,
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
        ties_method=cfg.SURVIVAL.TIES,
    )

    checkpoint = ModelCheckpoint(
        dirpath=Path(out_dir) / "checkpoints",
        filename="best",
        monitor="valid_c_index",
        mode="max",
        save_top_k=1,
    )
    callbacks = [checkpoint]
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
        model = WSISurvivalTrainer.load_from_checkpoint(checkpoint.best_model_path)

    predictions = [
        predict_split(model, _build_loader(datasets["train_eval"], cfg, shuffle=False), "train"),
        predict_split(model, val_loader, "val"),
        predict_split(model, test_loader, "test"),
    ]
    eval_times = list(cfg.SURVIVAL.EVAL_TIMES)
    metrics = {
        prediction.split: evaluate_predictions(prediction, predictions[0], eval_times) for prediction in predictions
    }

    save_survival_report(out_dir, predictions, metrics)
    export_attention(model, test_loader, Path(out_dir) / "attention", top_k=cfg.OUTPUT.TOP_K)
    logger.info("test metrics: %s", metrics["test"])
    return metrics


def run(cfg: CfgNode) -> dict:
    """Run the WSI survival pipeline described by ``cfg``.

    Args:
        cfg: Pipeline configuration, see :func:`kalecancer.config.get_cfg_defaults`.

    Returns:
        Metrics for the run; cross-validation additionally reports fold aggregates.

    Raises:
        PipelineError: If no patients match the configured inputs.
    """
    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    feature_root, clinical_path = resolve_dataset(cfg)
    bags, summary = build_cohort(
        feature_root=feature_root,
        clinical_path=clinical_path,
        endpoint=cfg.SURVIVAL.ENDPOINT,
        expected_dim=cfg.MODEL.INPUT_DIM,
        validate_features=cfg.DATASET.VALIDATE_FEATURES,
    )
    summary.log()
    write_json(out_dir / "cohort_summary.json", summary.as_dict())

    if not bags:
        raise PipelineError(
            f"no patients matched between {feature_root} and {clinical_path}; " f"see {out_dir / 'cohort_summary.json'}"
        )

    if cfg.DATASET.NUM_FOLDS:
        folds = stratified_patient_folds(
            bags, num_folds=cfg.DATASET.NUM_FOLDS, val_ratio=cfg.DATASET.VAL_RATIO, seed=cfg.SOLVER.SEED
        )
        fold_metrics = [run_split(split, cfg, out_dir / f"fold_{index}") for index, split in enumerate(folds)]
        cross_validated = summarise_folds(fold_metrics)
        metrics = {"cross_validated": cross_validated, "folds": fold_metrics}
        write_json(out_dir / "metrics.json", metrics)
        logger.info(
            "cross-validated test C-index: %.4f +/- %.4f",
            cross_validated["c_index"]["mean"],
            cross_validated["c_index"]["std"],
        )
    else:
        split = split_patients(
            bags,
            train_ratio=cfg.DATASET.TRAIN_RATIO,
            val_ratio=cfg.DATASET.VAL_RATIO,
            test_ratio=cfg.DATASET.TEST_RATIO,
            seed=cfg.SOLVER.SEED,
        )
        metrics = run_split(split, cfg, out_dir)

    logger.info("artefacts written to %s", out_dir)
    return metrics
