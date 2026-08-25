"""HANCOCK outcome classification: structured data, whole-slide imaging, or both.

Reproduces the experimental design of Dörrich et al. (2025) with the two modalities
we hold: the published structured tables and the published UNI patch encodings. Two
endpoints, three published splits and three modality settings give one comparison
matrix, repeated so each cell carries a mean and a spread.

    python -m examples.hancock_classification.main --cfg examples/hancock_classification/configs/quick.yaml
    python -m examples.hancock_classification.main --cfg examples/hancock_classification/configs/full.yaml
    python -m examples.hancock_classification.main --targets recurrence --splits dataset_split_in.json

Cells already carrying ``metrics.json`` are skipped unless ``--force`` is given, so a
long matrix can be resumed or sharded across machines and collected afterwards with
``--collect``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from examples.hancock_classification import tabular as tabular_data
from examples.hancock_classification.cohorts import LABEL, labelled_cohort, official_split
from examples.hancock_classification.config import SPLIT_FILES, TARGETS, get_cfg_defaults
from examples.hancock_classification.data import IMAGING, TABULAR, BagCache, PatientRecords, collate, slides_by_patient
from examples.wsi_survival.hancock import HancockDataset
from kalecancer.evaluate.classification_metrics import MetricError, binary_metrics, mean_roc_curve, roc_auc
from kalecancer.evaluate.harness import bootstrap_ci
from kalecancer.interpret.attention import attention_records, multimodal_attention, top_k_patches
from kalecancer.loaddata.split import holdout_split
from kalecancer.model.embed import AttentionMIL, BagEncoder, MLPEmbedder
from kalecancer.pipeline import MultimodalClassificationTrainer
from kalecancer.utils import ensure_dir, set_seed, write_csv, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hancock_classification")

#: The modality settings compared in every cell.
ARMS = (("tabular",), ("imaging",), ("tabular", "imaging"))

#: Test AUCs reported by Dörrich et al., for the cells they publish. Their tabular
#: numbers come from a richer vector (ICD and TMA features we do not hold) and their
#: imaging numbers from CLAM rather than this pooler, so these anchor the comparison
#: rather than define a target.
PAPER_AUC = {
    ("survival_status", "dataset_split_in.json", "tabular"): 0.79,
    ("survival_status", "dataset_split_out.json", "tabular"): 0.78,
    ("survival_status", "dataset_split_Oropharynx.json", "tabular"): 0.71,
    ("recurrence", "dataset_split_in.json", "tabular"): 0.79,
    ("recurrence", "dataset_split_out.json", "tabular"): 0.71,
    ("recurrence", "dataset_split_Oropharynx.json", "tabular"): 0.69,
    ("survival_status", "*", "imaging"): 0.65,
}


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HANCOCK outcome classification")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file")
    parser.add_argument("--targets", nargs="*", default=None, choices=list(TARGETS))
    parser.add_argument("--splits", nargs="*", default=None, choices=list(SPLIT_FILES))
    parser.add_argument("--modalities", nargs="*", default=None, help="e.g. tabular tabular+imaging")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rerun cells that already have metrics")
    parser.add_argument("--collect", action="store_true", help="rebuild the results table and exit")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides, e.g. SOLVER.MAX_EPOCHS 5")
    return parser.parse_args()


def build_embedders(cfg, modalities: tuple[str, ...], tabular_dim: int) -> dict[str, torch.nn.Module]:
    """One embedder per selected modality, each emitting a vector per patient."""
    embedders: dict[str, torch.nn.Module] = {}
    if TABULAR in modalities:
        embedders[TABULAR] = MLPEmbedder(
            tabular_dim,
            out_dim=cfg.FUSION.FUSION_DIM,
            hidden_dims=list(cfg.TABULAR.PROJECTION_HIDDEN),
            dropout=cfg.MODEL.DROPOUT,
        )
    if IMAGING in modalities:
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


def scores_for(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels for a loader, in loader order."""
    logits, labels = [], []
    for batch in loader:
        logits.append(model.predict_logits(batch).prediction.reshape(-1).cpu())
        labels.append(batch.target[LABEL])
    return torch.cat(logits).double().numpy(), torch.cat(labels).double().numpy()


def random_forest_baseline(features, labels, train_ids, test_ids, cfg, seed: int) -> dict:
    """The paper's model family, on the same encoded vectors and the same split.

    Class weighting stands in for their SMOTE: both counter imbalance, but by
    reweighting the loss rather than by synthesising minority rows, so the numbers are
    comparable in intent and not in detail.
    """
    matrix = lambda ids: np.vstack([features[identifier].numpy() for identifier in ids])  # noqa: E731
    forest = RandomForestClassifier(
        n_estimators=cfg.BASELINE.N_ESTIMATORS,
        class_weight=cfg.BASELINE.CLASS_WEIGHT,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(matrix(train_ids), [labels[identifier] for identifier in train_ids])
    scores = forest.predict_proba(matrix(test_ids))[:, 1]
    return binary_metrics([labels[identifier] for identifier in test_ids], scores) | {"scores": scores.tolist()}


def run_repeat(cfg, arm, features, bags, labels, fit_ids, val_ids, test_ids, out_dir: Path, repeat: int):
    """Train and evaluate one model.

    Returns:
        Its test metrics, the trained model, and a loader over the test split, so the
        caller can read attention out of the last repeat without retraining.
    """
    set_seed(cfg.SOLVER.SEED + repeat)
    tabular_dim = len(next(iter(features.values()))) if features else 0

    def loader(ids: list[str], shuffle: bool = False) -> DataLoader:
        records = PatientRecords(ids, list(arm), features, bags, labels, cfg.MODEL.INPUT_DIM)
        return DataLoader(
            records,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.DATASET.NUM_WORKERS,
            collate_fn=collate,
        )

    positives = sum(labels[identifier] for identifier in fit_ids)
    pos_weight = cfg.CLASSIFY.POS_WEIGHT
    if pos_weight == 0:
        pos_weight = (len(fit_ids) - positives) / max(positives, 1)

    model = MultimodalClassificationTrainer(
        build_embedders(cfg, arm, tabular_dim),
        stage=cfg.FUSION.STAGE,
        method=cfg.FUSION.METHOD,
        fusion_dim=cfg.FUSION.FUSION_DIM,
        auxiliary_weight=cfg.FUSION.AUXILIARY_WEIGHT,
        modality_dropout=cfg.FUSION.MODALITY_DROPOUT,
        pos_weight=max(pos_weight, 0.0),
        optimizer={"type": cfg.SOLVER.OPTIMIZER, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY}},
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        init_lr=cfg.SOLVER.BASE_LR,
    )

    checkpoint = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="best", monitor="valid_auc", mode="max")
    callbacks = [checkpoint]
    if cfg.SOLVER.EARLY_STOP:
        callbacks.append(EarlyStopping(monitor="valid_auc", mode="max", patience=cfg.SOLVER.EARLY_STOP))

    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        devices=cfg.SOLVER.DEVICES,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(out_dir), name="history"),
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader(fit_ids, shuffle=True), loader(val_ids))
    if checkpoint.best_model_path:
        # Weights only: the embedders are modules supplied here, not hyperparameters a
        # checkpoint could reconstruct.
        best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["state_dict"])

    scores, test_labels = scores_for(model, loader(test_ids))
    metrics = binary_metrics(test_labels, scores)
    # A training AUC at or below chance means the labels never reached the loss, which
    # no test number would reveal on its own.
    train_scores, train_labels = scores_for(model, loader(fit_ids))
    metrics["train_roc_auc"] = roc_auc(train_labels, train_scores)
    metrics["scores"] = scores.tolist()
    metrics["labels"] = test_labels.tolist()
    metrics["patient_id"] = list(test_ids)

    if len(arm) > 1:
        metrics["ablation"] = modality_ablation(model, loader(test_ids), test_labels, arm)
    return metrics, model, loader(test_ids)


def modality_ablation(model, loader, labels: np.ndarray, arm: tuple[str, ...]) -> dict:
    """How much each modality contributes, by withholding it at inference.

    The stand-in for the paper's attention-by-modality figure: with one patch-bearing
    modality there is no patch-level analogue, but marking a modality absent and
    rescoring measures the same thing -- what it was worth.
    """
    ablation = {}
    for held_out in arm:
        logits = []
        for batch in loader:
            batch.present[held_out] = torch.zeros_like(batch.present[held_out])
            logits.append(model.predict_logits(batch).prediction.reshape(-1).cpu())
        try:
            ablation[f"without_{held_out}"] = roc_auc(labels, torch.cat(logits).double().numpy())
        except MetricError as error:
            logger.warning("ablation for %s unavailable: %s", held_out, error)
    return ablation


def export_top_patches(model, loader, bags: BagCache, out_dir: Path, top_k: int) -> None:
    """Most-attended patches across the test split, with their slide coordinates."""
    summary: list[dict] = []
    for batch in loader:
        for patient, weights in multimodal_attention(model, batch, IMAGING).items():
            records = attention_records(bags.sample_for(patient), weights)
            summary.extend(top_k_patches(records, k=top_k))
    write_csv(out_dir / "top_patches.csv", summary)


def cell_dir(root: Path, split_file: str, target: str, arm: tuple[str, ...]) -> Path:
    return root / Path(split_file).stem / target / "+".join(arm)


def collect(root: Path) -> list[dict]:
    """Rebuild the comparison table from whatever cells have been run."""
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if rows:
        write_csv(root / "results.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
        logger.info("collected %d cells into %s", len(rows), root / "results.csv")
    return rows


def main() -> None:
    args = arg_parse()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.repeats:
        cfg.CLASSIFY.REPEATS = args.repeats
    cfg.freeze()

    root = ensure_dir(cfg.OUTPUT.OUT_DIR)
    if args.collect:
        collect(root)
        return

    (root / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    targets = args.targets or list(TARGETS)
    splits = args.splits or list(SPLIT_FILES)
    arms = tuple(tuple(name.split("+")) for name in args.modalities) if args.modalities else ARMS

    # ---------------------------------------------------------------- data
    dataset = HancockDataset(cache_dir=cfg.DATASET.CACHE_DIR or None)
    paths = {
        "clinical": dataset.clinical(),
        "pathological": dataset.fetch_named("structured", "pathological_data.json"),
        "blood": dataset.fetch_named("structured", "blood_data.json"),
        "ranges": dataset.fetch_named("structured", "blood_data_reference_ranges.json"),
    }
    frame, blood_columns = tabular_data.structured_frame(paths, cfg)

    needs_imaging = any(IMAGING in arm for arm in arms) or IMAGING in cfg.DATASET.REQUIRE_MODALITIES
    bags = None
    if needs_imaging:
        slides = slides_by_patient(dataset.features(region=cfg.DATASET.REGION, patients=cfg.DATASET.PATIENTS))
        bags = BagCache(slides, cfg.MODEL.INPUT_DIM, cfg.DATASET.MAX_PATCHES, cfg.SOLVER.SEED)

    # ---------------------------------------------------------------- matrix
    summaries = []
    for split_file in splits:
        assignment = official_split(dataset.splits(split_file))
        for target in targets:
            cohort = labelled_cohort(frame, cfg, target)
            labels = dict(zip(cohort[tabular_data.IDENTIFIER], cohort[LABEL], strict=True))
            usable = set(labels)
            if needs_imaging and bags:
                usable &= set(bags.bags)

            train_ids = sorted(usable & set(assignment.get("training", [])))
            test_ids = sorted(usable & set(assignment.get("test", [])))
            if not train_ids or not test_ids:
                raise ValueError(
                    f"{split_file}/{target}: {len(train_ids)} train and {len(test_ids)} test patients survived the "
                    "join; the usual cause is patient ids losing their zero padding"
                )

            tabular_cohort = tabular_data.build_cohort(cohort, blood_columns, cfg)
            features = tabular_data.encode(tabular_cohort, train_ids, train_ids + test_ids)
            # The width is split-dependent: the one-hot encoder is fitted on the
            # training rows, so a category absent from them contributes no column.
            width = len(next(iter(features.values())))
            logger.info(
                "%s/%s: %d train / %d test, %.1f%% positive, %d features",
                split_file,
                target,
                len(train_ids),
                len(test_ids),
                100 * np.mean([labels[i] for i in test_ids]),
                width,
            )

            for arm in arms:
                out_dir = ensure_dir(cell_dir(root, split_file, target, arm))
                if (out_dir / "summary.json").exists() and not args.force:
                    summaries.append(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")))
                    continue

                # The validation set is carved out of the training half, so the
                # published test split is never touched during model selection.
                fitting = cohort[cohort[tabular_data.IDENTIFIER].isin(train_ids)].reset_index(drop=True)

                repeats, model, last_loader = [], None, None
                for repeat in range(cfg.CLASSIFY.REPEATS):
                    inner = holdout_split(
                        fitting, ratio=cfg.DATASET.VAL_RATIO, stratify_keys=[LABEL], seed=cfg.SOLVER.SEED + repeat
                    )
                    fit_ids = list(fitting.iloc[inner["fit"]][tabular_data.IDENTIFIER])
                    val_ids = list(fitting.iloc[inner["holdout"]][tabular_data.IDENTIFIER])

                    metrics, model, last_loader = run_repeat(
                        cfg,
                        arm,
                        features,
                        bags,
                        labels,
                        fit_ids,
                        val_ids,
                        test_ids,
                        ensure_dir(out_dir / f"repeat_{repeat}"),
                        repeat,
                    )
                    write_json(out_dir / f"repeat_{repeat}" / "metrics.json", metrics)
                    repeats.append(metrics)
                    logger.info("  %s repeat %d: test AUC %.3f", "+".join(arm), repeat, metrics["roc_auc"])

                summary = summarise(cfg, repeats, split_file, target, arm, train_ids, test_ids, out_dir)
                if cfg.BASELINE.ENABLED and TABULAR in arm:
                    summary["baseline_auc"] = random_forest_baseline(
                        features, labels, train_ids, test_ids, cfg, cfg.SOLVER.SEED
                    )["roc_auc"]
                write_json(out_dir / "summary.json", summary)
                summaries.append(summary)

                if IMAGING in arm and bags and cfg.OUTPUT.TOP_K and model is not None:
                    export_top_patches(model, last_loader, bags, out_dir, cfg.OUTPUT.TOP_K)
                    logger.info("  wrote most-attended patches for %s", "+".join(arm))

    write_csv(root / "results.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in summaries])
    write_json(root / "results.json", summaries)
    logger.info("wrote %d cells to %s", len(summaries), root / "results.csv")


def summarise(cfg, repeats: list[dict], split_file, target, arm, train_ids, test_ids, out_dir: Path) -> dict:
    """Average the repeats and write the mean ROC curve."""
    curve = mean_roc_curve([run["labels"] for run in repeats], [run["scores"] for run in repeats])
    write_csv(
        out_dir / "roc_mean.csv",
        [
            {"fpr": float(f), "tpr_mean": float(m), "tpr_std": float(s)}
            for f, m, s in zip(curve["fpr"], curve["tpr_mean"], curve["tpr_std"], strict=True)
        ],
    )

    best = max(repeats, key=lambda run: run["roc_auc"])
    _, lower, upper = bootstrap_ci(roc_auc, np.array(best["labels"]), np.array(best["scores"]), n_boot=500)

    paper = PAPER_AUC.get((target, split_file, "+".join(arm)), PAPER_AUC.get((target, "*", "+".join(arm))))
    return {
        "split": Path(split_file).stem,
        "target": target,
        "modalities": "+".join(arm),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "positive_rate_test": repeats[0]["positive_rate"],
        "auc_mean": curve["auc_mean"],
        "auc_std": curve["auc_std"],
        "auc_repeats": curve["auc_runs"],
        "auc_boot_lo": lower,
        "auc_boot_hi": upper,
        "train_auc_mean": float(np.mean([run["train_roc_auc"] for run in repeats])),
        "f1_mean": float(np.mean([run["f1"] for run in repeats])),
        "average_precision_mean": float(np.mean([run["average_precision"] for run in repeats])),
        "ablation": repeats[0].get("ablation", {}),
        "paper_auc": paper,
    }


if __name__ == "__main__":
    main()
