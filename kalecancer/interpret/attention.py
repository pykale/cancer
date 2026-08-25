"""Attention-based interpretation of WSI survival predictions.

Attention weights say which patches drove a patient's predicted risk. Each weight is
exported with the coordinate and source slide of its patch, so heatmaps can be
rendered later against the original whole-slide image. No image is produced here:
raw slides are not part of this pipeline, and drawing a heatmap without them would
mean inventing the underlying tissue.

As with any attribution on a Cox model, high attention marks regions associated with
*higher predicted risk*, not a calibrated probability of death.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kalecancer.utils.io import ensure_dir, write_csv


def attention_records(sample: dict, attention: torch.Tensor) -> list[dict]:
    """Join a bag's attention weights to its patch coordinates.

    Args:
        sample: The bag the attention was computed for.
        attention: ``(num_patches,)`` weights, aligned with ``sample["features"]``.

    Returns:
        One record per patch with its slide, coordinate and attention weight.

    Raises:
        ValueError: If the attention length does not match the number of patches.
    """
    attention = attention.detach().cpu().flatten()
    num_patches = len(sample["coords"])
    if len(attention) != num_patches:
        raise ValueError(
            f"attention has {len(attention)} weights but the bag has {num_patches} patches; "
            "they must stay aligned for coordinates to be meaningful"
        )

    coords = sample["coords"].cpu()
    slide_index = sample["slide_index"].cpu()
    return [
        {
            "patient_id": sample["group_id"],
            "slide_id": sample["slide_ids"][int(slide_index[i])],
            "x": int(coords[i, 0]),
            "y": int(coords[i, 1]),
            "attention": float(attention[i]),
        }
        for i in range(len(attention))
    ]


def top_k_patches(records: list[dict], k: int = 10) -> list[dict]:
    """The ``k`` highest-attention patches, most attended first.

    Raises:
        ValueError: If ``k`` is negative, which would otherwise trim the most
            attended patches instead of selecting them.
    """
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")
    return sorted(records, key=lambda record: record["attention"], reverse=True)[:k]


def export_attention(
    model,
    loader: DataLoader,
    out_dir: str | Path,
    top_k: int = 20,
) -> Path:
    """Export per-patch attention for every patient in ``loader``.

    Writes one ``<patient_id>.csv`` of patch-level attention per patient, plus a
    ``top_patches.csv`` summary of the most attended patches across the cohort.

    Args:
        model: A trained :class:`~kalecancer.pipeline.WSISurvivalTrainer`.
        loader: Loader yielding collated bags.
            Use a loader without patch subsampling so attention covers whole slides.
        out_dir: Directory for the exported files.
        top_k: Number of top patches to summarise per patient.

    Returns:
        The directory written to.
    """
    out_dir = ensure_dir(out_dir)
    summary: list[dict] = []

    for batch in loader:
        _, attentions = model.predict_risk(batch)
        for sample, attention in zip(batch["samples"], attentions, strict=True):
            records = attention_records(sample, attention)
            write_csv(out_dir / f"{sample['group_id']}.csv", records)
            summary.extend(top_k_patches(records, k=top_k))

    write_csv(out_dir / "top_patches.csv", summary)
    return out_dir


def multimodal_attention(model, batch, modality: str) -> dict[str, torch.Tensor]:
    """Per-patch attention for one bag modality of a fused model.

    A bag-pooling embedder such as :class:`~kalecancer.model.embed.BagEncoder` keeps
    its last attention weights; this pairs them with the patients they came from.
    Unlike :func:`export_attention`, which serves the single-modality WSI trainer,
    this reads a :class:`~kalecancer.loaddata.sample.PatientBatch` and so works with
    the multimodal trainers.

    Args:
        model: A trainer exposing ``model.embedders`` and a gradient-free predict.
        batch: The batch to run, carrying ``patient_id``.
        modality: Which modality's embedder to read attention from.

    Returns:
        Attention weights keyed by patient id, one vector per patient.

    Raises:
        KeyError: If the model carries no such modality.
        AttributeError: If that modality's embedder records no attention.
    """
    embedders = model.model.embedders
    if modality not in embedders:
        raise KeyError(f"model has no modality {modality!r}; available: {sorted(embedders)}")

    predict = getattr(model, "predict_logits", None) or model.predict_risk
    predict(batch)

    embedder = embedders[modality]
    if not hasattr(embedder, "last_attention"):
        raise AttributeError(
            f"the {modality!r} embedder is a {type(embedder).__name__}, which records no attention; "
            "only a bag-pooling embedder such as BagEncoder does"
        )
    return dict(zip(batch.patient_id, embedder.last_attention, strict=True))


def collect_attention(model, batch: dict) -> dict[str, list[dict]]:
    """Attention records for one batch, keyed by patient id."""
    _, attentions = model.predict_risk(batch)
    return {
        sample["group_id"]: attention_records(sample, attention)
        for sample, attention in zip(batch["samples"], attentions, strict=True)
    }
