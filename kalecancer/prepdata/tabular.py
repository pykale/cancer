"""Fitted preprocessing for tabular cohorts.

A :class:`TabularPreprocessor` is the artifact
:meth:`~kalecancer.loaddata.tabular.TabularCohort.fit_preprocessor` produces: a
fitted ``ColumnTransformer`` plus the provenance needed to police it. It belongs
to exactly one cross-validation fold and is never mutated after fitting.

Keeping it separate from the cohort is what makes fold discipline checkable rather
than merely conventional. "Which statistics did fold 3 apply, and to whose rows
were they fitted" is a question you can ask this object; it is not a question you
can ask a dataset that fitted itself in place.
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

    Args:
        transformer (ColumnTransformer | None): The fitted transformer, or ``None``
            when the cohort declared no stateful transforms and columns pass
            through as they are.
        columns (list[str]): Feature columns to select, in order. Stored because a
            ``ColumnTransformer`` must be given the same frame layout it saw during
            fitting.
        feature_names (dict[str, list[str]]): Column names *after* encoding, keyed
            by modality. Not the same as ``columns`` once a one-hot encoder is
            involved. A dict of one entry here rather than a flat list, because
            that is the shape :class:`~kalecancer.loaddata.protocols.Preprocessor`
            declares and a composite preprocessor needs -- merging one-key dicts is
            trivial, reconciling flat lists is not.
        fitted_on (frozenset[str]): Identifiers whose statistics are baked into
            this artifact.

    Note:
        ``fitted_on`` is empty when ``transformer`` is ``None``. That is not an
        oversight: a passthrough bakes in no statistics, so no row's information is
        carried in it, and the leakage guard should not fire on one. The set means
        "whose data is inside this object", which for a passthrough is nobody's.
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
        """Apply these transforms to a frame's rows.

        Args:
            frame (pd.DataFrame): Rows to transform. Must contain :attr:`columns`.

        Returns:
            Tensor: ``(n_rows, n_features)``, float32.

        Note:
            An empty frame short-circuits. An empty fold is a legitimate thing to
            build -- a cross-validation harness may produce one -- and scikit-learn
            refuses zero rows with "Found array with 0 sample(s)", which says
            nothing about what the caller actually did.
        """
        if frame.empty:
            return torch.zeros((0, self.width), dtype=torch.float32)
        block = frame[self.columns]
        values = block.to_numpy() if self.transformer is None else self.transformer.transform(block)
        # np.array, not asarray: forces a writable copy, because torch warns when
        # asked to wrap a read-only view. Casting straight to float32 rather than
        # via float64 avoids a second full copy of the matrix.
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
