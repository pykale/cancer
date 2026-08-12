"""WSI primary-tumour survival prediction on precomputed UNI patch features.

Runs the full workflow: discover feature files, match them to clinical survival
labels at patient level, build leakage-safe splits, train attention MIL with a Cox
head, evaluate, and export attention for interpretation.

Examples:
    python examples/wsi_survival/main.py
    python examples/wsi_survival/main.py --cfg configs/hancock_primary_tumour.yaml
    python examples/wsi_survival/main.py DATASET.FEATURE_ROOT /data/features SOLVER.MAX_EPOCHS 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
from config import get_cfg_defaults
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from kalecancer.evaluate import evaluate_predictions, predict_split, save_survival_report, summarise_folds
from kalecancer.interpret import export_attention
from kalecancer.loaddata import (
    CohortSplit,
    WSIFeatureBagDataset,
    build_cohort,
    collate_bags,
    split_patients,
    stratified_patient_folds,
)
from kalecancer.pipeline import WSISurvivalTrainer
from kalecancer.utils import ensure_dir, seed_worker, set_seed, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("wsi_survival")


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI primary-tumour survival prediction")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file", type=str)
    parser.add_argument("--devices", default="auto", help="devices passed to the Lightning trainer")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="config overrides as KEY VALUE pairs, e.g. SOLVER.MAX_EPOCHS 5",
    )
    return parser.parse_args()


def build_loader(dataset: WSIFeatureBagDataset, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_bags,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.DATASET.NUM_WORKERS > 0,
    )


def build_datasets(split: CohortSplit, cfg) -> dict[str, WSIFeatureBagDataset]:
    """Training subsamples patches to bound memory; evaluation keeps whole bags.

    ``train_eval`` re-reads the training patients without subsampling, so the baseline
    hazard and censoring weights are estimated from the same full-bag predictions the
    evaluated splits produce.
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


def run_fold(split: CohortSplit, cfg, out_dir: Path, devices) -> dict:
    """Train and evaluate one train/validation/test split."""
    logger.info("split sizes: %s", split.sizes())
    datasets = build_datasets(split, cfg)

    train_loader = build_loader(datasets["train"], cfg, shuffle=True)
    val_loader = build_loader(datasets["val"], cfg, shuffle=False)
    test_loader = build_loader(datasets["test"], cfg, shuffle=False)

    model = WSISurvivalTrainer(
        input_dim=cfg.MODEL.INPUT_DIM,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        attention_dim=cfg.MODEL.ATTENTION_DIM,
        dropout=cfg.MODEL.DROPOUT,
        gated=cfg.MODEL.GATED,
        optimizer={
            "type": cfg.SOLVER.OPTIMIZER,
            "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        },
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
        ties_method=cfg.SURVIVAL.TIES,
    )

    checkpoint = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
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
        devices=devices,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(out_dir), name="history"),
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)
    if checkpoint.best_model_path:
        model = WSISurvivalTrainer.load_from_checkpoint(checkpoint.best_model_path)

    predictions = [
        predict_split(model, build_loader(datasets["train_eval"], cfg, shuffle=False), "train"),
        predict_split(model, val_loader, "val"),
        predict_split(model, test_loader, "test"),
    ]
    train_predictions = predictions[0]
    eval_times = list(cfg.SURVIVAL.EVAL_TIMES)
    metrics = {
        prediction.split: evaluate_predictions(prediction, train_predictions, eval_times) for prediction in predictions
    }

    save_survival_report(out_dir, predictions, metrics)
    export_attention(model, test_loader, out_dir / "attention", top_k=cfg.OUTPUT.TOP_K)
    logger.info("test metrics: %s", metrics["test"])
    return metrics


def main() -> None:
    args = arg_parse()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    out_dir = ensure_dir(cfg.OUTPUT.OUT_DIR)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    bags, summary = build_cohort(
        feature_root=cfg.DATASET.FEATURE_ROOT,
        clinical_path=cfg.DATASET.CLINICAL_PATH,
        endpoint=cfg.SURVIVAL.ENDPOINT,
        expected_dim=cfg.MODEL.INPUT_DIM,
        validate_features=cfg.DATASET.VALIDATE_FEATURES,
    )
    summary.log()
    write_json(out_dir / "cohort_summary.json", summary.as_dict())

    if not bags:
        raise SystemExit("no patients matched; check DATASET.FEATURE_ROOT and DATASET.CLINICAL_PATH")

    if cfg.DATASET.NUM_FOLDS:
        folds = stratified_patient_folds(
            bags, num_folds=cfg.DATASET.NUM_FOLDS, val_ratio=cfg.DATASET.VAL_RATIO, seed=cfg.SOLVER.SEED
        )
        fold_metrics = [
            run_fold(split, cfg, ensure_dir(out_dir / f"fold_{index}"), args.devices)
            for index, split in enumerate(folds)
        ]
        cross_validated = summarise_folds(fold_metrics)
        write_json(out_dir / "metrics.json", {"cross_validated": cross_validated, "folds": fold_metrics})
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
        run_fold(split, cfg, out_dir, args.devices)

    logger.info("artefacts written to %s", out_dir)


if __name__ == "__main__":
    main()
