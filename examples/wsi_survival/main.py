"""WSI primary-tumour survival prediction on precomputed UNI patch features.

Runs the full workflow: discover feature files, match them to clinical survival labels
at patient level, build leakage-safe splits, train attention MIL with a Cox head,
evaluate, and export attention for interpretation.

Examples:
    python examples/wsi_survival/main.py --cfg configs/hancock_primary_tumour.yaml
    python examples/wsi_survival/main.py --cfg configs/hancock_primary_tumour.yaml SOLVER.MAX_EPOCHS 5
"""

from __future__ import annotations

import argparse
import logging

from config import get_cfg_defaults
from hancock import fetch_for

from kalecancer.loaddata.clinical_access import endpoint_from_config
from kalecancer.pipeline.wsi_survival_runner import run

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

    # The dataset belongs to this experiment; the runner only needs paths and an
    # endpoint, both described by the configuration.
    run(cfg, endpoint=endpoint_from_config(cfg), fetch=lambda: fetch_for(cfg))


if __name__ == "__main__":
    main()
