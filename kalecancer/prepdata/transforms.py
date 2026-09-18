"""Transforms for table modalities, fitted on the training patients by the Pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


class ColumnGroup:
    """Columns that share a sequence of sklearn steps."""

    def __init__(self, columns: list[str], steps: list[TransformerMixin]):
        self.columns = columns
        self.steps = steps


class TableTransform(TransformerMixin, BaseEstimator):
    """A config-friendly ``ColumnTransformer``: every input column must be in one group or in ``drop``.

    jsonargparse cannot validate the untyped ``transformers`` tuples of ``ColumnTransformer``, so a typo
    inside a nested step would pass silently in a YAML config. Here every step is typed.
    """

    def __init__(self, groups: dict[str, ColumnGroup], drop: list[str]):
        self.groups = groups
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: Any = None) -> TableTransform:
        listed = [c for group in self.groups.values() for c in group.columns]
        duplicated = sorted({c for c in listed if listed.count(c) > 1})
        unlisted = [c for c in X.columns if c not in set(listed) | set(self.drop)]
        unknown = sorted((set(listed) | set(self.drop)) - set(X.columns))
        if duplicated or unlisted or unknown:
            raise ValueError(
                "TableTransform: every column must be in exactly one group or in drop; "
                f"unlisted={unlisted}, duplicated={duplicated}, unknown={unknown}"
            )
        steps = {
            name: make_pipeline(*[clone(step) for step in group.steps]) if group.steps else "passthrough"
            for name, group in self.groups.items()
        }
        self.transformer_ = ColumnTransformer(
            [(name, steps[name], group.columns) for name, group in self.groups.items()],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.transformer_.get_feature_names_out()


def _check_no_dropped_columns(transform: TransformerMixin, modality: str) -> None:
    if not isinstance(transform, ColumnTransformer) or transform.remainder != "drop":
        return
    for label, _, columns in transform.transformers_:
        if label == "remainder" and len(columns):
            names = [transform.feature_names_in_[c] if isinstance(c, int | np.integer) else c for c in columns]
            raise ValueError(
                f"transforms[{modality!r}]: the ColumnTransformer silently drops columns {list(names)}; transform "
                "them, remove them from the table, or set remainder='passthrough'"
            )
