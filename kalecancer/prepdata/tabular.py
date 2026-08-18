"""Fitted preprocessing for tabular cohorts.

The artifact ``TabularCohort.fit_preprocessor`` produces: a fitted
``ColumnTransformer`` plus the provenance needed to police it. Belongs to one fold
and is never mutated after fitting.

Keeping it off the cohort is what makes "which statistics did fold 3 apply, and to
whose rows were they fitted" a question you can actually ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch import Tensor


@dataclass(frozen=True)
class TabularPreprocessor:
    """Fitted transforms for one fold, with the rows they were fitted on.

    ``fitted_on`` is empty for a passthrough. Not an oversight: it means "whose data is
    inside this object", which for a passthrough is nobody's, so the leakage guard
    should not fire on one.

    Args:
        transformer (ColumnTransformer | None): ``None`` when no stateful transform was
            declared and columns pass through.
        columns (list[str]): Feature columns to select, in order. A
            ``ColumnTransformer`` needs the frame layout it was fitted on.
        feature_names (dict[str, list[str]]): Names *after* encoding, keyed by modality
            -- the shape ``Preprocessor`` declares, so composites merge trivially.
        fitted_on (frozenset[str]): Identifiers whose statistics are baked in.
    """

    transformer: ColumnTransformer | None
    columns: list[str]
    feature_names: dict[str, list[str]] = field(default_factory=dict)
    fitted_on: frozenset[str] = frozenset()

    @property
    def width(self) -> int:
        """Number of columns the transformed matrix has."""
        return sum(len(names) for names in self.feature_names.values())

    def transform(self, frame: pd.DataFrame) -> Tensor:
        """Apply these transforms to a frame's rows, returning ``(n_rows, n_features)`` float32.

        An empty frame short-circuits: an empty fold is legitimate, and scikit-learn
        would refuse it with a message that says nothing about what the caller did.
        """
        if frame.empty:
            return torch.zeros((0, self.width), dtype=torch.float32)
        block = frame[self.columns]
        values = block.to_numpy() if self.transformer is None else self.transformer.transform(block)
        # np.array, not asarray: torch warns on a read-only view, so force a copy.
        return torch.from_numpy(np.array(values, dtype=np.float32))

    def describe(self) -> str:
        """Human-readable summary of what this fold applied."""
        if self.transformer is None:
            return f"passthrough ({len(self.columns)} columns, nothing fitted)"
        blocks = []
        for label, pipeline, columns in self.transformer.transformers_:
            if pipeline == "passthrough":
                rendered = "passthrough"
            elif pipeline == "drop":
                continue
            else:
                rendered = " -> ".join(type(step).__name__ for _, step in pipeline.steps)
            blocks.append(f"  {label:<12}({len(columns)} cols): {rendered}")
        head = f"fitted on {len(self.fitted_on)} rows -> {self.width} features"
        return "\n".join([head, *blocks])
