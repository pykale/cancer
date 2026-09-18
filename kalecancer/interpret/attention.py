"""Attention export: per-instance attention weights joined with the modality's coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from torch import Tensor, nn

from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.model.models import StageList

if TYPE_CHECKING:
    from kalecancer.pipeline.pipeline import Pipeline


def attention(pipeline: Pipeline, data: MultimodalDataset, modality: str, branch: str | None = None) -> pd.DataFrame:
    """Per-instance attention weights of the stage that exposes ``attention``, joined with the modality's
    coordinates."""
    source = data.modalities.get(modality)
    if not callable(getattr(source, "coords", None)):
        raise TypeError(f"modality {modality!r} does not provide coords")
    assert source is not None
    single, stages = pipeline._single_modality(data, modality, branch)
    hits = [k for k, stage in enumerate(stages) if callable(getattr(stage, "attention", None))]
    if len(hits) != 1:
        raise ValueError(f"{modality!r} needs exactly one stage with attention(), found {len(hits)}")
    before, attending = StageList(list(stages)[: hits[0]], modality), stages[hits[0]]

    def step(model: nn.Module, batch: dict) -> tuple[list[str], list[Tensor]]:
        x = before(batch["inputs"][modality], batch["ids"])
        return batch["ids"], [w.float().cpu() for w in attending.attention(x)]

    frames = []
    for ids, weights in pipeline._infer(single, step):
        for pid, w in zip(ids, weights, strict=True):
            coords = source.coords(pid)
            if len(coords) != len(w):
                raise ValueError(f"{pid}: {len(w)} attention weights for {len(coords)} coordinates")
            frames.append(coords.assign(attention=w.numpy()).assign(patient_id=pid))
    table = pd.concat(frames, ignore_index=True)
    return table[["patient_id", *[c for c in table.columns if c not in ("patient_id", "attention")], "attention"]]
