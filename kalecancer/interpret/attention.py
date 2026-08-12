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

from kalecancer.loaddata.wsi_dataset import BagBatch, BagSample
from kalecancer.utils.io import ensure_dir, write_csv


def attention_records(sample: BagSample, attention: torch.Tensor) -> list[dict]:
    """Join a bag's attention weights to its patch coordinates.

    Args:
        sample: The bag the attention was computed for.
        attention: ``(num_patches,)`` weights, aligned with ``sample.features``.

    Returns:
        One record per patch with its slide, coordinate and attention weight.

    Raises:
        ValueError: If the attention length does not match the number of patches.
    """
    attention = attention.detach().cpu().flatten()
    if len(attention) != len(sample.coords):
        raise ValueError(
            f"attention has {len(attention)} weights but the bag has {len(sample.coords)} patches; "
            "they must stay aligned for coordinates to be meaningful"
        )

    coords = sample.coords.cpu()
    slide_index = sample.slide_index.cpu()
    return [
        {
            "patient_id": sample.patient_id,
            "slide_id": sample.slide_ids[int(slide_index[i])],
            "x": int(coords[i, 0]),
            "y": int(coords[i, 1]),
            "attention": float(attention[i]),
        }
        for i in range(len(attention))
    ]


def top_k_patches(records: list[dict], k: int = 10) -> list[dict]:
    """The ``k`` highest-attention patches, most attended first."""
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
        loader: Loader yielding :class:`~kalecancer.loaddata.wsi_dataset.BagBatch`.
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
        for sample, attention in zip(batch.samples, attentions, strict=True):
            records = attention_records(sample, attention)
            write_csv(out_dir / f"{sample.patient_id}.csv", records)
            summary.extend(top_k_patches(records, k=top_k))

    write_csv(out_dir / "top_patches.csv", summary)
    return out_dir


def collect_attention(model, batch: BagBatch) -> dict[str, list[dict]]:
    """Attention records for one batch, keyed by patient id."""
    _, attentions = model.predict_risk(batch)
    return {
        sample.patient_id: attention_records(sample, attention)
        for sample, attention in zip(batch.samples, attentions, strict=True)
    }
