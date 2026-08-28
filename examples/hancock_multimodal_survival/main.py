"""HANCOCK multimodal survival: clinical, imaging, or both, with a Cox head.

One code path serves three runs; ``DATASET.MODALITIES`` decides which. With two
modalities, ``FUSION.STAGE`` and ``FUSION.METHOD`` select how they are combined,
so comparing intermediate against late fusion is a configuration change.

The time-to-event half of a pair: ``examples/hancock_multimodal_classification``
runs the same fusion over the same two modalities against a binary endpoint, and
differs from this only in the task it passes to the trainer.

    python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/tabular.yaml
    python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/imaging.yaml
    python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/multimodal.yaml
    python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/multimodal.yaml FUSION.STAGE late
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from examples.hancock import HancockDataset, official_split
from examples.hancock.clinical import (
    endpoint_from_config,
    load_clinical_records,
    survival_table,
)
from examples.hancock_multimodal_survival.config import get_cfg_defaults
from examples.hancock_multimodal_survival.data import (
    CLINICAL,
    IMAGING,
    build_tabular_cohort,
    embed_clinical,
    slides_by_patient,
)
from kalecancer.evaluate.survival_metrics import concordance_index, time_dependent_auc, usable_eval_times
from kalecancer.evaluate.survival_predictions import summarise_folds
from kalecancer.loaddata import ColumnTarget, FeatureBagSource, ModalitySource, MultimodalDataset, VectorSource
from kalecancer.loaddata.multimodal_access import collate_ragged
from kalecancer.loaddata.splitting import CohortSplitter, CrossValidation, Predefined
from kalecancer.model.embed import BagEncoder, MLPEmbedder
from kalecancer.model.layers import AttentionMIL
from kalecancer.pipeline import CohortTrainer, SurvivalTask
from kalecancer.utils import ensure_dir, set_seed, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hancock_multimodal_survival")


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HANCOCK multimodal survival")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, e.g. FUSION.STAGE late")
    return parser.parse_args()


def build_embedders(cfg, clinical_dim: int) -> dict[str, torch.nn.Module]:
    """One embedder per selected modality, each emitting a vector per patient."""
    embedders: dict[str, torch.nn.Module] = {}
    if CLINICAL in cfg.DATASET.MODALITIES:
        # The frozen TabICL representation is projected here; this is what trains.
        embedders[CLINICAL] = MLPEmbedder(
            clinical_dim,
            out_dim=cfg.FUSION.FUSION_DIM,
            hidden_dims=list(cfg.TABULAR.PROJECTION_HIDDEN),
            dropout=cfg.MODEL.DROPOUT,
        )
    if IMAGING in cfg.DATASET.MODALITIES:
        embedders[IMAGING] = BagEncoder(
            AttentionMIL(
                input_dim=cfg.MODEL.INPUT_DIM,
                hidden_dim=cfg.MODEL.HIDDEN_DIM,
                attention_dim=cfg.MODEL.ATTENTION_DIM,
                dropout=cfg.MODEL.DROPOUT,
                gated=cfg.MODEL.GATED,
            )
        )
    return embedders


def evaluate(model, loader, train_loader, cfg) -> dict:
    """Risk scores for a split, scored with the shared survival metrics."""

    def risks(source) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, times, events = [], [], []
        for batch in source:
            scores.append(model.predict(batch).prediction.reshape(-1).cpu())
            times.append(batch.target["time"])
            events.append(batch.target["event"])
        return torch.cat(scores), torch.cat(times), torch.cat(events)

    risk, time, event = risks(loader)
    risk_np, time_np, event_np = risk.double().numpy(), time.double().numpy(), event.bool().numpy()
    metrics: dict = {
        "num_patients": int(len(risk)),
        "num_events": int(event_np.sum()),
        "c_index": concordance_index(risk_np, time_np, event_np),
    }

    train_risk, train_time, train_event = risks(train_loader)
    horizons = usable_eval_times(torch.tensor(cfg.SURVIVAL.EVAL_TIMES).numpy(), time_np)
    horizons = usable_eval_times(horizons, train_time.double().numpy())
    if horizons.size >= 2:
        try:
            auc, mean_auc = time_dependent_auc(
                train_time.double().numpy(), train_event.bool().numpy(), time_np, event_np, risk_np, horizons
            )
            metrics["auc"] = {f"{t:.0f}": float(v) for t, v in zip(horizons, auc, strict=True)}
            metrics["mean_auc"] = float(mean_auc)
        except ValueError as error:
            logger.warning("time-dependent AUC unavailable: %s", error)
    return metrics


def run_split(cfg, split, usable_table, clinical_path, slides, labels, modalities, out_dir: Path) -> dict:
    """Train and evaluate one train/validation/test split.

    Everything fitted on data -- the TabICL context as much as the model weights --
    is built inside here from this split's training rows alone. That is what makes a
    cross-validated run honest: a context built once outside the loop would carry
    every fold's test patients into every other fold's representation.
    """

    def ids_for(key: str) -> list[str]:
        return list(usable_table.iloc[split[key]]["patient_id"])

    fit_ids, val_ids, test_ids = ids_for("train"), ids_for("val"), ids_for("test")

    clinical = (
        embed_clinical(build_tabular_cohort(clinical_path, cfg), fit_ids, fit_ids + val_ids + test_ids, cfg)
        if CLINICAL in modalities
        else {}
    )

    target = ColumnTarget(labels, columns={"time": "duration", "event": "event"})

    def loader(ids: list[str], shuffle: bool = False, cap: int | None = None) -> DataLoader:
        # One source per modality, so imaging-only, clinical-only and both are the
        # same call with a different dictionary.
        sources: dict[str, ModalitySource] = {}
        if CLINICAL in modalities:
            sources[CLINICAL] = VectorSource(clinical)
        if IMAGING in modalities:
            sources[IMAGING] = FeatureBagSource(
                slides, feature_dim=cfg.MODEL.INPUT_DIM, max_patches=cap, seed=cfg.SOLVER.SEED
            )

        return DataLoader(
            MultimodalDataset(ids, sources, target=target),
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.DATASET.NUM_WORKERS,
            collate_fn=collate_ragged,
        )

    cap = cfg.DATASET.MAX_PATCHES or None
    model = CohortTrainer(
        build_embedders(cfg, len(next(iter(clinical.values()))) if clinical else 0),
        task=SurvivalTask(),
        stage=cfg.FUSION.STAGE,
        method=cfg.FUSION.METHOD,
        fusion_dim=cfg.FUSION.FUSION_DIM,
        auxiliary_weight=cfg.FUSION.AUXILIARY_WEIGHT,
        modality_dropout=cfg.FUSION.MODALITY_DROPOUT,
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
    )

    checkpoint = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="best", monitor="valid_c_index", mode="max")
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
    trainer.fit(model, loader(fit_ids, shuffle=True, cap=cap), loader(val_ids))
    if checkpoint.best_model_path:
        # Weights only: the embedders are modules supplied here, not hyperparameters
        # a checkpoint could reconstruct.
        best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["state_dict"])

    # The training split supplies the IPCW weights and the baseline hazard, so it is
    # passed separately from the split being scored.
    return {
        "train_patients": len(fit_ids),
        "val_patients": len(val_ids),
        "test_patients": len(test_ids),
        "test": evaluate(model, loader(test_ids), loader(fit_ids), cfg),
    }


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
    modalities = list(cfg.DATASET.MODALITIES)
    logger.info("modalities: %s | fusion: %s/%s", modalities, cfg.FUSION.STAGE, cfg.FUSION.METHOD)

    # ---------------------------------------------------------------- data
    dataset = HancockDataset(cache_dir=cfg.DATASET.CACHE_DIR or None)
    clinical_path = dataset.clinical()

    labels, exclusions = survival_table(load_clinical_records(clinical_path), endpoint=endpoint_from_config(cfg))
    required = set(modalities) | set(cfg.DATASET.REQUIRE_MODALITIES)
    # Slide features are several gigabytes, so they are only fetched when needed.
    slides = (
        slides_by_patient(dataset.features(region=cfg.DATASET.REGION, patients=cfg.DATASET.PATIENTS))
        if IMAGING in required
        else {}
    )

    usable = set(labels["patient_id"])
    if IMAGING in required:
        usable &= set(slides)
    usable_table = labels[labels["patient_id"].isin(usable)].reset_index(drop=True)

    # --------------------------------------------------------------- splits
    split_options = {"stratify_by": ["event"], "val_size": cfg.DATASET.VAL_RATIO, "random_state": cfg.SOLVER.SEED}
    if cfg.DATASET.SPLIT_MODE == "cv":
        splitter: CohortSplitter = CrossValidation(n_splits=cfg.DATASET.NUM_FOLDS, **split_options)
        splits = list(splitter.split(usable_table))
        logger.info("%d-fold cross-validation over %d usable patients", len(splits), len(usable_table))
    elif cfg.DATASET.SPLIT_MODE == "published":
        assignment = official_split(dataset.splits(cfg.DATASET.SPLIT_FILE))
        splitter = Predefined(assignment, **split_options)
        splits = list(splitter.split(usable_table))
        logger.info(
            "published split %s: %d train / %d test of %d usable patients | excluded %s",
            cfg.DATASET.SPLIT_FILE,
            len(splits[0]["train"]) + len(splits[0]["val"]),
            len(splits[0]["test"]),
            len(usable_table),
            {reason: len(ids) for reason, ids in exclusions.items() if ids},
        )
    else:
        raise ValueError(f"unknown DATASET.SPLIT_MODE {cfg.DATASET.SPLIT_MODE!r}; expected 'published' or 'cv'")

    # ------------------------------------------------------- train and score
    fold_metrics = [
        run_split(
            cfg,
            split,
            usable_table,
            clinical_path,
            slides,
            labels,
            modalities,
            ensure_dir(out_dir / f"fold_{index}") if len(splits) > 1 else out_dir,
        )
        for index, split in enumerate(splits)
    ]

    metrics: dict = {
        "split_mode": cfg.DATASET.SPLIT_MODE,
        "split": Path(cfg.DATASET.SPLIT_FILE).stem if cfg.DATASET.SPLIT_MODE == "published" else None,
        "modalities": modalities,
        "fusion_stage": cfg.FUSION.STAGE if len(modalities) > 1 else "none",
        "fusion_method": cfg.FUSION.METHOD if len(modalities) > 1 else "none",
        "folds": fold_metrics,
    }
    if len(fold_metrics) > 1:
        # summarise_folds reads each fold's "test" entry itself.
        metrics["cross_validated"] = summarise_folds(fold_metrics)
        logger.info("cross-validated test metrics: %s", metrics["cross_validated"])
    else:
        logger.info("test metrics: %s", fold_metrics[0]["test"])

    write_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
