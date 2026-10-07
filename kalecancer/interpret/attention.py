"""Attention export: per-instance attention weights joined with the modality's description of each instance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from torch import Tensor

from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import BagModality

if TYPE_CHECKING:
    from kalecancer.model.models import StageList
    from kalecancer.pipeline.pipeline import Pipeline


def attention(pipeline: Pipeline, data: MultimodalDataset, modality: str, branch: str | None = None) -> pd.DataFrame:
    """Per-instance attention weights of the stage that exposes ``attention``, joined with the modality's
    ``instances``. The stages before it must keep each bag's instances one to one with the loaded rows."""
    if modality not in data.modalities:
        raise ValueError(f"the dataset has no modality {modality!r}")
    source = data.modalities[modality]
    if not isinstance(source, BagModality):
        raise TypeError(f"Attention export needs a BagModality. '{modality}' is a {type(source).__name__}")

    def step(stages: StageList, x: Any, ids: list[str]) -> list[Tensor]:
        hits = [k for k, stage in enumerate(stages) if callable(getattr(stage, "attention", None))]
        if len(hits) != 1:
            raise ValueError(f"{modality!r} needs exactly one stage with attention(), found {len(hits)}")
        # attention() is an optional stage method, which the typed nn.Module.__getattr__ does not know about
        attending: Any = stages[hits[0]]
        stages.check_input(x, ids)  # the stages run one at a time below, so StageList.forward never checks
        for k, stage in enumerate(list(stages)[: hits[0]]):
            sizes = [len(bag) for bag in x]
            x = stage(x, ids) if getattr(stage, "needs_ids", False) else stage(x)
            if [len(bag) for bag in x] != sizes:
                raise ValueError(
                    f"encoding[{modality!r}][{k}] {type(stage).__name__} changes the number of instances in a bag, "
                    "so attention weights cannot be matched to its instances; sample instances only in training mode, "
                    "or select them in the modality"
                )
        return [w.float().cpu() for w in attending.attention(x)]

    frames = []
    for ids, weights in pipeline.run_modality(data, modality, step, branch):
        for pid, w in zip(ids, weights, strict=True):
            rows = source.instances(pid)
            if len(rows) != len(w):
                raise ValueError(f"{pid}: {len(w)} attention weights for {len(rows)} instances")
            frames.append(rows.assign(attention=w.numpy()).assign(patient_id=pid))
    table = pd.concat(frames, ignore_index=True)
    return table[["patient_id", *[c for c in table.columns if c not in ("patient_id", "attention")], "attention"]]
