"""HANCOCK multimodal survival: clinical, imaging, or both.

One code path serves three runs; ``DATASET.MODALITIES`` decides which. With two
modalities, ``FUSION.STAGE`` and ``FUSION.METHOD`` select how they are combined,
so comparing intermediate against late fusion is a configuration change.

    python examples/hancock_multimodal/main.py --cfg configs/tabular.yaml
    python examples/hancock_multimodal/main.py --cfg configs/imaging.yaml
    python examples/hancock_multimodal/main.py --cfg configs/multimodal.yaml
    python examples/hancock_multimodal/main.py --cfg configs/multimodal.yaml FUSION.STAGE late
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# Appended, not inserted: this directory stays first so the local `config` module
# is not shadowed by the WSI example's.
sys.path.append(str(Path(__file__).resolve().parents[1] / "wsi_survival"))

from config import get_cfg_defaults  # noqa: E402
from data import (  # noqa: E402
    CLINICAL,
    IMAGING,
    MultimodalCohort,
    build_tabular_cohort,
    collate,
    embed_clinical,
    official_split,
    slides_by_patient,
)
from hancock import HancockDataset  # noqa: E402

from kalecancer.evaluate.survival_metrics import time_dependent_auc, usable_eval_times  # noqa: E402
from kalecancer.loaddata.clinical_access import (  # noqa: E402
    endpoint_from_config,
    load_clinical_records,
    survival_table,
)
from kalecancer.loaddata.split import holdout_split  # noqa: E402
from kalecancer.model.embed import AttentionMIL, BagEncoder, MLPEmbedder  # noqa: E402
from kalecancer.pipeline import MultimodalSurvivalTrainer  # noqa: E402
from kalecancer.survival.metrics import concordance_index  # noqa: E402
from kalecancer.utils import ensure_dir, set_seed, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hancock_multimodal")


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
            scores.append(model.predict_risk(batch).prediction.reshape(-1).cpu())
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
    split_path = dataset.splits()

    labels, exclusions = survival_table(load_clinical_records(clinical_path), endpoint=endpoint_from_config(cfg))
    labels = labels.rename(columns={"duration": "time"})
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
    published = official_split(split_path)
    train_ids = sorted(usable & set(published.get("training", [])))
    test_ids = sorted(usable & set(published.get("test", [])))
    logger.info(
        "cohort: %d usable patients | official split %d train / %d test | excluded %s",
        len(usable),
        len(train_ids),
        len(test_ids),
        {reason: len(ids) for reason, ids in exclusions.items() if ids},
    )

    clinical = (
        embed_clinical(build_tabular_cohort(clinical_path, cfg), train_ids, train_ids + test_ids, cfg)
        if CLINICAL in modalities
        else {}
    )

    # Validation comes out of the training half, so the published test set stays untouched.
    fit_table = labels[labels["patient_id"].isin(train_ids)].reset_index(drop=True)
    inner = holdout_split(
        fit_table, ratio=cfg.DATASET.VAL_RATIO, stratify_keys=["event"], seed=cfg.SOLVER.SEED
    )
    fit_ids = list(fit_table.iloc[inner["fit"]]["patient_id"])
    val_ids = list(fit_table.iloc[inner["holdout"]]["patient_id"])

    def loader(ids: list[str], shuffle: bool = False, cap: int | None = None) -> DataLoader:
        source = MultimodalCohort(
            ids, modalities, clinical, slides, labels, cfg.MODEL.INPUT_DIM, cap, cfg.SOLVER.SEED
        )
        return DataLoader(
            source,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.DATASET.NUM_WORKERS,
            collate_fn=collate,
        )

    cap = cfg.DATASET.MAX_PATCHES or None
    # --------------------------------------------------------------- model
    model = MultimodalSurvivalTrainer(
        build_embedders(cfg, len(next(iter(clinical.values()))) if clinical else 0),
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
    trainer.fit(model, loader(fit_ids, shuffle=True, cap=cap), loader(val_ids))
    if checkpoint.best_model_path:
        # Weights only: the embedders are modules supplied here, not hyperparameters
        # a checkpoint could reconstruct.
        best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["state_dict"])

    # ---------------------------------------------------------- evaluation
    metrics = {
        "modalities": modalities,
        "fusion_stage": cfg.FUSION.STAGE if len(modalities) > 1 else "none",
        "fusion_method": cfg.FUSION.METHOD if len(modalities) > 1 else "none",
        "train_patients": len(fit_ids),
        "test": evaluate(model, loader(test_ids), loader(fit_ids), cfg),
    }
    write_json(out_dir / "metrics.json", metrics)
    logger.info("test metrics: %s", metrics["test"])


if __name__ == "__main__":
    main()
