"""Attention-based interpretation of bag-pooled predictions.

Attention weights say which patches drove a patient's prediction. Each weight is
exported with the coordinate and source slide of its patch, so heatmaps can be
rendered later against the original whole-slide image. No image is produced here:
raw slides are not part of this pipeline, and drawing a heatmap without them would
mean inventing the underlying tissue.

Attribution marks regions associated with a *higher predicted score* -- higher risk
for a Cox model, higher probability for a classifier -- never a calibrated
probability of the outcome itself.

Everything here reads a :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`: the
weights come from the modality's embedder, and the coordinates they are joined to
come from the batch's ``metadata``, which is why that field exists.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.utils.io import ensure_dir, write_csv


def bag_attention(model, batch: PatientBatch, modality: str) -> dict[str, torch.Tensor]:
    """Per-patch attention for one bag modality, keyed by patient.

    A bag-pooling embedder such as :class:`~kalecancer.model.embed.BagEncoder` keeps
    the weights from its last forward pass; this runs the batch and pairs them with
    the patients they came from.

    Args:
        model: A trainer exposing ``embedders`` and a gradient-free ``predict``.
        batch: The batch to run, carrying ``patient_id``.
        modality: Which modality's embedder to read attention from.

    Returns:
        Attention weights keyed by patient id, one vector per patient.

    Raises:
        KeyError: If the model carries no such modality.
        AttributeError: If that modality's embedder records no attention.
    """
    embedders = model.embedders
    if modality not in embedders:
        raise KeyError(f"model has no modality {modality!r}; available: {sorted(embedders)}")

    model.predict(batch)

    embedder = embedders[modality]
    if not hasattr(embedder, "last_attention"):
        raise AttributeError(
            f"the {modality!r} embedder is a {type(embedder).__name__}, which records no attention; "
            "only a bag-pooling embedder such as BagEncoder does"
        )
    return dict(zip(batch.patient_id, embedder.last_attention, strict=True))


def attention_records(
    patient_id: str,
    attention: torch.Tensor,
    coords: torch.Tensor,
    slide_index: torch.Tensor,
    slide_ids: tuple[str, ...],
) -> list[dict]:
    """Join one bag's attention weights to its patch coordinates.

    Args:
        patient_id: Whose bag this is.
        attention: ``(num_patches,)`` weights, aligned with the bag's patches.
        coords: ``(num_patches, 2)`` patch coordinates.
        slide_index: ``(num_patches,)`` index into ``slide_ids``.
        slide_ids: Slides contributing to the bag.

    Returns:
        One record per patch with its slide, coordinate and attention weight.

    Raises:
        ValueError: If the attention length does not match the number of patches.
    """
    attention = attention.detach().cpu().flatten()
    if len(attention) != len(coords):
        raise ValueError(
            f"attention has {len(attention)} weights but the bag has {len(coords)} patches; "
            "they must stay aligned for coordinates to be meaningful"
        )

    coords = coords.cpu()
    slide_index = slide_index.cpu()
    return [
        {
            "patient_id": patient_id,
            "slide_id": slide_ids[int(slide_index[i])],
            "x": int(coords[i, 0]),
            "y": int(coords[i, 1]),
            "attention": float(attention[i]),
        }
        for i in range(len(attention))
    ]


def batch_records(model, batch: PatientBatch, modality: str) -> dict[str, list[dict]]:
    """Attention records for one batch, keyed by patient id.

    Raises:
        KeyError: If the batch carries no coordinate metadata for its bags, which is
            what the weights would be joined to.
    """
    missing = {"coords", "slide_index", "slide_ids"} - set(batch.metadata)
    if missing:
        raise KeyError(
            f"batch metadata is missing {sorted(missing)}; attention can only be placed on a slide "
            "when the dataset carries the coordinates its patches came from"
        )

    weights = bag_attention(model, batch, modality)
    return {
        patient_id: attention_records(
            patient_id,
            weights[patient_id],
            batch.metadata["coords"][index],
            batch.metadata["slide_index"][index],
            batch.metadata["slide_ids"][index],
        )
        for index, patient_id in enumerate(batch.patient_id)
    }


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
    modality: str = "wsi",
    top_k: int = 20,
) -> Path:
    """Export per-patch attention for every patient in ``loader``.

    Writes one ``<patient_id>.csv`` of patch-level attention per patient, plus a
    ``top_patches.csv`` summary of the most attended patches across the cohort.

    Args:
        model: A trained trainer carrying a bag-pooling embedder for ``modality``.
        loader: Loader yielding batches with coordinate metadata. Use a loader
            without patch subsampling so attention covers whole slides.
        out_dir: Directory for the exported files.
        modality: Which modality's attention to export.
        top_k: Number of top patches to summarise per patient.

    Returns:
        The directory written to.
    """
    out_dir = ensure_dir(out_dir)
    summary: list[dict] = []

    for batch in loader:
        for patient_id, records in batch_records(model, batch, modality).items():
            write_csv(out_dir / f"{patient_id}.csv", records)
            summary.extend(top_k_patches(records, k=top_k))

    write_csv(out_dir / "top_patches.csv", summary)
    return out_dir
