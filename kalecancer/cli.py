"""Command-line interface for running KaleCancer pipelines.

Wraps the same configuration objects the Python API uses, so a run started from the
command line and one started from a YAML file are the same run. Presets replace the
most common configuration edits::

    kalecancer wsi-survival --features <dir> --clinical <file> --preset quick
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("kalecancer")

#: Named configurations covering the common experiments. Values are config overrides.
PRESETS: dict[str, dict] = {
    "quick": {
        "DATASET.MAX_PATCHES": 256,
        "DATASET.VALIDATE_FEATURES": False,
        "MODEL.HIDDEN_DIM": 128,
        "MODEL.ATTENTION_DIM": 64,
        "SOLVER.MAX_EPOCHS": 3,
        "SOLVER.BATCH_SIZE": 32,
        "SOLVER.EARLY_STOP": 0,
    },
    "default": {},
    "cv": {"DATASET.NUM_FOLDS": 5},
    "dss": {"SURVIVAL.ENDPOINT": "DSS", "SOLVER.BATCH_SIZE": 32},
}

PRESET_HELP = {
    "quick": "3 epochs on subsampled patches, for checking a setup end to end",
    "default": "single seeded train/validation/test split",
    "cv": "5-fold patient-level cross-validation",
    "dss": "disease-specific survival endpoint",
}


def _add_wsi_survival_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "wsi-survival",
        help="survival prediction from precomputed whole-slide patch features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", type=str, help="directory of HDF5 patch features")
    parser.add_argument("--clinical", type=str, help="JSON file of clinical records")
    parser.add_argument("--out", type=str, help="directory for results")
    parser.add_argument(
        "--source",
        choices=["local"],
        help="where the data comes from",
    )
    parser.add_argument("--patients", type=int, help="patients to fetch from a remote source; 0 fetches all")
    parser.add_argument("--region", choices=["primary", "lymph_node"], help="anatomical region to fetch")
    parser.add_argument("--cache-dir", type=str, help="cache for fetched data")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="; ".join(f"{name}: {text}" for name, text in PRESET_HELP.items()),
    )
    parser.add_argument("--epochs", type=int, help="training epochs")
    parser.add_argument("--batch-size", type=int, help="patients per batch")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--folds", type=int, help="cross-validation folds; 0 uses a single split")
    parser.add_argument("--endpoint", choices=["OS", "DSS"], help="survival endpoint")
    parser.add_argument("--cfg", type=str, help="YAML config file applied before the options above")
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="further overrides as KEY VALUE pairs, e.g. MODEL.DROPOUT 0.1"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kalecancer", description="Multimodal cancer AI for the PyKale ecosystem")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_wsi_survival_parser(subparsers)
    return parser


def resolve_config(args: argparse.Namespace):
    """Merge preset, file, and flags into one configuration.

    Precedence, lowest first: defaults, ``--preset``, ``--cfg`` file, named flags,
    then trailing ``KEY VALUE`` overrides.
    """
    from kalecancer.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_list([item for key, value in PRESETS[args.preset].items() for item in (key, value)])

    if args.cfg:
        cfg.merge_from_file(args.cfg)

    flags = {
        "DATASET.SOURCE": args.source,
        "DATASET.FEATURE_ROOT": args.features,
        "DATASET.CLINICAL_PATH": args.clinical,
        "DATASET.REGION": args.region,
        "DATASET.PATIENTS": args.patients,
        "DATASET.CACHE_DIR": args.cache_dir,
        "OUTPUT.OUT_DIR": args.out,
        "SOLVER.MAX_EPOCHS": args.epochs,
        "SOLVER.BATCH_SIZE": args.batch_size,
        "SOLVER.SEED": args.seed,
        "DATASET.NUM_FOLDS": args.folds,
        "SURVIVAL.ENDPOINT": args.endpoint,
    }
    # Supplying explicit paths implies the local source unless stated otherwise.
    if args.source is None and (args.features or args.clinical):
        flags["DATASET.SOURCE"] = "local"
    cfg.merge_from_list([item for key, value in flags.items() if value is not None for item in (key, value)])

    if args.opts:
        cfg.merge_from_list(args.opts)
    return cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    if args.print_config:
        print(cfg.dump())  # noqa: T201 - stdout is this command's output
        return 0

    if cfg.DATASET.SOURCE == "local":
        missing = [
            flag
            for flag, path in (("--features", cfg.DATASET.FEATURE_ROOT), ("--clinical", cfg.DATASET.CLINICAL_PATH))
            if not path or not Path(path).exists()
        ]
        if missing:
            logger.error(
                "path not found for %s; pass an existing path or use --cfg",
                ", ".join(missing),
            )
            return 2

    from kalecancer.loaddata.clinical_access import endpoint_from_config
    from kalecancer.loaddata.dataset_access import DatasetAccessError
    from kalecancer.pipeline.wsi_survival_runner import PipelineError, run

    cfg.freeze()
    try:
        run(cfg, endpoint=endpoint_from_config(cfg))
    except (PipelineError, DatasetAccessError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
