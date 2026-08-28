"""Render the run's artefacts as the paper's figures.

Plotting lives here rather than in ``kalecancer`` on purpose: the library exports
coordinates and metrics so the caller decides how to present them, which is the same
convention :mod:`kalecancer.interpret.attention` follows. Nothing here recomputes
anything -- every figure is read back from what ``main.py`` already wrote, so this is
cheap to rerun and cannot disagree with the reported numbers.

    python -m examples.hancock_multimodal_classification.figures
    python -m examples.hancock_multimodal_classification.figures --out-dir outputs/hancock_multimodal_classification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in a container or over SSH; write files instead.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hancock_figures")

SPLIT_ORDER = ["dataset_split_in", "dataset_split_out", "dataset_split_Oropharynx"]
SPLIT_LABEL = {
    "dataset_split_in": "In distribution",
    "dataset_split_out": "Out of distribution",
    "dataset_split_Oropharynx": "Oropharynx held out",
}
SPLIT_SHORT = {"dataset_split_in": "in", "dataset_split_out": "out", "dataset_split_Oropharynx": "Oropharynx"}
ARM_ORDER = ["tabular", "imaging", "tabular+imaging"]
ARM_COLOUR = {"tabular": "#2c7fb8", "imaging": "#d95f0e", "tabular+imaging": "#31a354"}
TARGETS = ["survival_status", "recurrence"]


def _style(axis, title: str = "") -> None:
    axis.set_title(title, fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def roc_grid(root: Path, out: Path) -> Path:
    """Mean ROC per split and endpoint, with a one-standard-deviation band.

    The paper's Fig 2E. The band is run-to-run variability across repeats, not a
    confidence interval on the AUC.
    """
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for row, target in enumerate(TARGETS):
        for column, split in enumerate(SPLIT_ORDER):
            axis = axes[row][column]
            for arm in ARM_ORDER:
                curve_path = root / split / target / arm / "roc_mean.csv"
                summary_path = root / split / target / arm / "summary.json"
                if not curve_path.exists():
                    continue
                curve = pd.read_csv(curve_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                colour = ARM_COLOUR[arm]
                axis.plot(
                    curve["fpr"],
                    curve["tpr_mean"],
                    color=colour,
                    linewidth=1.6,
                    label=f"{arm} {summary['auc_mean']:.3f}±{summary['auc_std']:.3f}",
                )
                axis.fill_between(
                    curve["fpr"],
                    curve["tpr_mean"] - curve["tpr_std"],
                    np.minimum(curve["tpr_mean"] + curve["tpr_std"], 1.0),
                    color=colour,
                    alpha=0.15,
                )
            axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=0.9)
            _style(axis, f"{SPLIT_LABEL[split]}\n{target}" if row == 0 else target)
            axis.legend(fontsize=7, loc="lower right", frameon=False)
            if column == 0:
                axis.set_ylabel("True positive rate", fontsize=9)
            if row == 1:
                axis.set_xlabel("False positive rate", fontsize=9)

    figure.suptitle("Mean ROC over 5 repeats, shaded ±1 SD  (cf. Dörrich et al., Fig 2E)", fontsize=11)
    figure.tight_layout()
    return _save(figure, out / "fig2e_roc_curves.png")


def umap_grid(root: Path, out: Path) -> Path:
    """The cohort projected to two dimensions, coloured by outcome and by split.

    The paper's Fig 2B-D. The projection is fitted on the whole cohort, so it
    describes the data rather than demonstrating that the classes separate.
    """
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for row, colour_by in enumerate(["label", "dataset"]):
        for column, split in enumerate(SPLIT_ORDER):
            axis = axes[row][column]
            path = root / split / "survival_status" / "umap.csv"
            if not path.exists():
                continue
            points = pd.read_csv(path, dtype={"patient_id": str})
            groups = (
                [(0, "living", "#4575b4"), (1, "deceased", "#d73027")]
                if colour_by == "label"
                else [("training", "training", "#bdbdbd"), ("test", "test", "#7b3294")]
            )
            for value, name, colour in groups:
                subset = points[points[colour_by] == value]
                axis.scatter(
                    subset["x"], subset["y"], s=7, alpha=0.7, c=colour, label=f"{name} ({len(subset)})", linewidths=0
                )
            _style(axis, SPLIT_LABEL[split] if row == 0 else "")
            axis.legend(fontsize=7, frameon=False, markerscale=1.6)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel("by outcome" if colour_by == "label" else "by split", fontsize=9)

    figure.suptitle("UMAP of the structured patient vectors  (cf. Dörrich et al., Fig 2B-D)", fontsize=11)
    figure.tight_layout()
    return _save(figure, out / "fig2bd_umap.png")


def modality_comparison(root: Path, out: Path) -> Path:
    """Per-repeat AUC by modality, against the values the paper reports.

    The analogue of their Fig 3E, which compares WSI, TMA and WSI+TMA.
    """
    # Read the per-cell summaries rather than results.csv: the per-repeat AUCs are a
    # list, and the flat table drops list-valued columns.
    cells = {}
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        cells[summary["split"], summary["target"], summary["modalities"]] = summary

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for axis, target in zip(axes, TARGETS, strict=True):
        positions, ticks = [], []
        for index, split in enumerate(SPLIT_ORDER):
            for offset, arm in enumerate(ARM_ORDER):
                summary = cells.get((split, target, arm))
                if summary is None:
                    continue
                repeats = summary["auc_repeats"]
                position = index * 4 + offset
                axis.boxplot(
                    repeats,
                    positions=[position],
                    widths=0.62,
                    patch_artist=True,
                    boxprops={"facecolor": ARM_COLOUR[arm], "alpha": 0.55, "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.2},
                    flierprops={"markersize": 3},
                )
                if summary.get("paper_multimodal_auc") is not None and arm == "tabular+imaging":
                    axis.plot(
                        position, summary["paper_multimodal_auc"], marker="*", markersize=12, color="black", zorder=5
                    )
                positions.append(position)
            ticks.append((index * 4 + 1, SPLIT_SHORT[split]))

        axis.axhline(0.5, ls="--", color="0.6", linewidth=0.9)
        if target == "survival_status":
            # Their WSI-only result is one number for the whole cohort, not one per
            # split, so it is a reference line rather than a per-split marker.
            axis.axhline(0.65, ls=":", color="#d95f0e", linewidth=1.3)
            axis.text(
                0.02,
                0.655,
                "paper, WSI-only MIL (not split-specific)",
                fontsize=6.5,
                color="#d95f0e",
                transform=axis.get_yaxis_transform(),
            )
        axis.set_xticks([t[0] for t in ticks])
        axis.set_xticklabels([t[1] for t in ticks], fontsize=9)
        axis.set_xlabel("published split", fontsize=9)
        _style(axis, target)
        axis.set_ylim(0.45, 0.85)
    axes[0].set_ylabel("Test ROC-AUC", fontsize=9)
    # One bar swatch per arm, plus a star marker for the published reference point.
    handles: list[Artist] = [plt.Rectangle((0, 0), 1, 1, fc=ARM_COLOUR[a], alpha=0.55) for a in ARM_ORDER]
    handles.append(plt.Line2D([], [], marker="*", ls="", color="black", markersize=12))
    axes[0].legend(
        handles, [*ARM_ORDER, "paper, multimodal vector"], fontsize=7.5, frameon=False, loc="lower left", ncol=2
    )

    figure.suptitle("Test AUC by modality, 5 repeats per cell  (cf. Dörrich et al., Fig 3E)", fontsize=11)
    figure.tight_layout()
    return _save(figure, out / "fig3e_modality_comparison.png")


def ablation_plot(root: Path, out: Path) -> Path:
    """What each modality was worth, by withholding it at inference.

    Stands in for their Fig 3F, which traces attention back to its source modality.
    With one patch-bearing modality there is no patch-level analogue, but withholding
    a modality and rescoring measures the same quantity.
    """
    rows = []
    for path in sorted(root.rglob("tabular+imaging/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        ablation = summary.get("ablation") or {}
        if ablation:
            rows.append(
                {
                    "cell": f"{SPLIT_LABEL[summary['split']].split()[0]}\n{summary['target']}",
                    "full": summary["auc_mean"],
                    "without tabular": ablation.get("without_tabular", np.nan),
                    "without imaging": ablation.get("without_imaging", np.nan),
                }
            )
    frame = pd.DataFrame(rows)

    figure, axis = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(frame))
    for offset, (column, colour) in enumerate(
        [("full", "#31a354"), ("without tabular", "#d95f0e"), ("without imaging", "#2c7fb8")]
    ):
        axis.bar(x + (offset - 1) * 0.27, frame[column], width=0.26, label=column, color=colour, alpha=0.85)
    axis.axhline(0.5, ls="--", color="0.6", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels(frame["cell"], fontsize=8)
    axis.set_ylabel("Test ROC-AUC", fontsize=9)
    axis.set_ylim(0.45, 0.8)
    axis.legend(fontsize=8, frameon=False, ncol=3)
    _style(axis, "Leave-one-modality-out on the fused model  (cf. Dörrich et al., Fig 3F)")

    figure.tight_layout()
    return _save(figure, out / "fig3f_modality_ablation.png")


def attention_maps(root: Path, out: Path, patients: int = 6) -> Path:
    """Where the imaging model looked, in slide coordinates.

    The nearest we can get to their Fig 3C. They show the most-attended tiles as
    images; the raw slides are not part of this pipeline, so only the coordinates and
    weights of the attended patches are drawn. No tissue is invented.
    """
    path = root / "dataset_split_in" / "survival_status" / "imaging" / "top_patches.csv"
    if not path.exists():
        raise FileNotFoundError(f"no attention export at {path}; run an imaging arm first")
    records = pd.read_csv(path, dtype={"patient_id": str})

    chosen = records["patient_id"].value_counts().head(patients).index
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    for axis, patient in zip(axes.ravel(), chosen, strict=False):
        subset = records[records["patient_id"] == patient]
        # Normalised weight, so panels are comparable: the raw value depends on how
        # many patches the bag held, since attention sums to one across the bag.
        share = subset["attention"] / subset["attention"].max()
        points = axis.scatter(
            subset["x"], subset["y"], c=share, s=70, cmap="inferno", vmin=0, vmax=1, linewidths=0.4, edgecolors="0.3"
        )
        _style(axis, f"patient {patient}  ({len(subset)} most attended)")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(
        points, ax=axes, fraction=0.03, pad=0.02, label="attention, relative to the patient's strongest patch"
    )

    figure.suptitle(
        "Most-attended patches in slide coordinates  (cf. Dörrich et al., Fig 3C)\n"
        "Positions and weights only: the source slides are outside this pipeline",
        fontsize=10,
    )
    return _save(figure, out / "fig3c_attention_maps.png")


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    logger.info("wrote %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the HANCOCK classification figures")
    parser.add_argument(
        "--out-dir", default="outputs/hancock_multimodal_classification", help="directory main.py wrote to"
    )
    args = parser.parse_args()

    root = Path(args.out_dir)
    if not (root / "results.csv").exists():
        raise SystemExit(f"no results in {root}; run examples.hancock_multimodal_classification.main first")

    figures = root / "figures"
    for render in (roc_grid, umap_grid, modality_comparison, ablation_plot, attention_maps):
        try:
            render(root, figures)
        except (FileNotFoundError, KeyError, ValueError) as error:
            logger.warning("skipping %s: %s", render.__name__, error)
    logger.info("figures written to %s", figures)


if __name__ == "__main__":
    main()
