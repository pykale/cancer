"""HANCOCK whole-slide survival prediction on precomputed UNI patch features.

Runs the full workflow: discover feature files, match them to clinical survival labels
at patient level, build leakage-safe splits, train attention MIL with a Cox head,
evaluate, and export attention for interpretation.

The imaging-only end of the survival examples. Adding the clinical table to this same
Cox objective is ``examples/hancock_multimodal_survival``.

Examples:
    python -m examples.hancock_wsi_survival.main --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour.yaml
    python -m examples.hancock_wsi_survival.main --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour.yaml SOLVER.MAX_EPOCHS 5
"""

from __future__ import annotations

import argparse
import logging

from examples.hancock import fetch_for, split_for
from examples.hancock.clinical import endpoint_from_config
from examples.hancock_wsi_survival.config import get_cfg_defaults
from examples.hancock_wsi_survival.runner import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI primary-tumour survival prediction")
    parser.add_argument("--cfg", default=None, help="path to a YAML config file", type=str)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="config overrides as KEY VALUE pairs, e.g. SOLVER.MAX_EPOCHS 5",
    )
    return parser.parse_args()


def main() -> None:
    args = arg_parse()
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # The dataset belongs to this experiment; the runner only needs paths, an
    # endpoint and the published assignment, all described by the configuration.
    run(
        cfg,
        endpoint=endpoint_from_config(cfg),
        fetch=lambda: fetch_for(cfg),
        splits=lambda: split_for(cfg),
    )


if __name__ == "__main__":
    main()
