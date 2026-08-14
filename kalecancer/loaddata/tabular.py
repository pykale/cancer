from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone as sk_clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from torch import Tensor

from kalecancer.loaddata.base import BaseDataset, NotFittedError
from kalecancer.survival.survival_target import SurvivalTarget

_READERS = {
    ".csv": lambda p, dtype: pd.read_csv(p, dtype=dtype),
    ".tsv": lambda p, dtype: pd.read_csv(p, sep="\t", dtype=dtype),
    ".json": lambda p, dtype: pd.read_json(p, dtype=dtype),
}


def _read_table(path: Path, id_column: str) -> pd.DataFrame:
    """Read a tabular file into a DataFrame, dispatching on suffix."""
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported table format '{path.suffix}'. Expected one of {sorted(_READERS)}.")
    return _prepare_frame(reader(path, {id_column: str}), id_column)


def _prepare_frame(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Normalise a freshly-sourced table: fresh row index, identifiers as strings.

    The identifier is forced to string because pandas infers ``"001"`` as the
    integer ``1`` and the leading zeros are gone -- invisible until a slide manifest
    keyed on ``"001"`` joins against it and silently matches nothing.

    Note that this can only *keep* an identifier a string, never recover one: by the
    time an in-memory frame arrives here, a caller who read it without
    ``dtype={id_column: str}`` has already lost the padding.
    """
    frame = frame.reset_index(drop=True)
    if id_column in frame.columns:
        frame[id_column] = frame[id_column].astype(str)
    return frame


class TabularDataset(BaseDataset):
    """Covariates read from a flat table, keyed by a sample identifier.

    Column roles are declared, and preprocessing is declared per role using plain
    scikit-learn transformers. Nothing is fitted at construction: transforms are
    fitted per cross-validation fold via :meth:`fit_on`, which returns a new
    instance rather than mutating this one, so a fold can never see statistics
    computed on data outside it.

    Args:
        source (str | Path | pd.DataFrame): A table to read (``.csv``, ``.tsv``,
            ``.json``), or an already-loaded ``DataFrame``. Anything pandas can read but
            this cannot -- parquet, Excel, a database query -- goes through the frame:
            read it yourself and pass the result. The frame is copied, so later edits to
            the caller's object do not reach the dataset.
            Read it with ``dtype={identifier: str}`` -- by the time a frame arrives
            here, a zero-padded ``"001"`` inferred as ``1`` is already unrecoverable.
        identifier (str): Column holding sample identifiers. Must be unique.
        target (SurvivalTarget | None, optional): Supervision target. ``None`` for a
            pure feature provider used as a component of a larger cohort.
        continuous (Sequence[str], optional): Numeric feature columns.
        categorical (Sequence[str], optional): Categorical feature columns.
        continuous_transform (optional): A scikit-learn transformer, or a list of them
            applied in order, for the continuous columns. ``None`` (the default) passes
            them through untouched. There is deliberately no default pipeline: imputation
            and scaling are modelling decisions that belong in the caller's script, where
            they can be read, reviewed and reported.
        categorical_transform (optional): As above, for the categorical columns.
        name (str, optional): Key for this modality's features in the item dict.
            Defaults to ``"clinical"``.

    Example:
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
        >>> cohort = TabularDataset(
        ...     "data/my_cohort.csv",
        ...     identifier="patient_id",
        ...     target=SurvivalTarget(time="os_days", event="vital_status",
        ...                           event_value="dead"),
        ...     continuous=["age", "bmi"],
        ...     continuous_transform=[SimpleImputer(strategy="median"), StandardScaler()],
        ...     categorical=["sex", "stage"],
        ...     categorical_transform=OneHotEncoder(handle_unknown="ignore",
        ...                                         sparse_output=False),
        ... )
        >>> train, test = cohort.split(test_size=0.2, random_state=0)
        >>> fitted_train = train.fit_on(range(len(train)))
        >>> fitted_test = fitted_train.transform(test)

        Passing a frame is the seam for anything that has to happen between reading
        the file and binding the target -- resolving rows the target refuses to
        guess at, or joining a second table -- so the decision stays visible in the
        script rather than in a separately-prepared file:

        >>> frame = pd.read_csv("data/my_cohort.csv", dtype={"patient_id": str})
        >>> frame = frame[frame.vital_status.notna()]   # 24 patients, status not recorded
        >>> cohort = TabularDataset(frame, identifier="patient_id", target=target)
    """

    target: SurvivalTarget | None

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        identifier: str,
        target: SurvivalTarget | None = None,
        continuous: Sequence[str] = (),
        categorical: Sequence[str] = (),
        continuous_transform: Any = None,
        categorical_transform: Any = None,
        name: str = "clinical",
    ):
        # Set before super().__init__, which triggers _load_index().
        self.identifier = identifier
        self.continuous = list(continuous)
        self.categorical = list(categorical)
        self.continuous_transform = continuous_transform
        self.categorical_transform = categorical_transform

        # Doubles as the in-memory source: seeded here when the caller passed a frame,
        # replaced by _load_index with the table read from self.path otherwise. The empty
        # placeholder is never read -- _load_index overwrites it before anything else runs.
        self._frame: pd.DataFrame = source.copy() if isinstance(source, pd.DataFrame) else pd.DataFrame()
        self._preprocessor_spec: ColumnTransformer | None = None
        self._preprocessor: ColumnTransformer | None = None
        self._matrix: Tensor | None = None
        self.feature_names: list[str] = []

        super().__init__(path=None if isinstance(source, pd.DataFrame) else source, name=name, target=target)

    def _has_index_source(self) -> bool:
        """Always: ``source`` is either a path to read or a frame already in hand."""
        return True

    # ------------------------------------------------------------------ #
    # index loading
    # ------------------------------------------------------------------ #

    def _load_index(self) -> None:
        """Read the table, validate the declared columns, bind the target.

        Reads the whole table because a clinical table is small; for this modality
        index and payload are the same read. The expensive-payload case that
        motivates the split is WSI, where this would read a manifest only.
        """
        self._frame = (
            _prepare_frame(self._frame, self.identifier)
            if self.path is None
            else _read_table(self.path, self.identifier)
        )
        self._validate_columns()

        self.identifiers = self._frame[self.identifier].tolist()
        duplicates = self._frame[self.identifier].duplicated()
        if duplicates.any():
            raise ValueError(
                f"Column '{self.identifier}' must be unique; found "
                f"{int(duplicates.sum())} duplicate(s), e.g. "
                f"{self._frame.loc[duplicates, self.identifier].unique()[:5].tolist()}"
            )

        if self.target is not None:
            self.target.bind(self._frame, self.identifier)

        self._preprocessor_spec = self._build_preprocessor()
        self._check_untransformed_roles()
        if self._preprocessor_spec is None:
            # Nothing to fit, so values are available immediately.
            self._materialise()

    def _validate_columns(self) -> None:
        columns = set(self._frame.columns)
        if self.identifier not in columns:
            raise ValueError(f"Identifier column '{self.identifier}' not found. Available: {sorted(columns)}")
        missing = [c for c in self.feature_columns if c not in columns]
        if missing:
            raise ValueError(f"Feature column(s) {missing} not found. Available: {sorted(columns)}")
        overlap = set(self.continuous) & set(self.categorical)
        if overlap:
            raise ValueError(f"Column(s) {sorted(overlap)} declared as both continuous and categorical.")
        if self.identifier in self.feature_columns:
            raise ValueError(f"Identifier column '{self.identifier}' cannot also be a feature.")

    def _build_preprocessor(self) -> ColumnTransformer | None:
        """Assemble a ColumnTransformer from the declared roles, or None if no-op."""
        blocks, stateful = [], False
        for columns, spec, label in self._roles():
            if not columns:
                continue
            steps = self._resolve(spec)
            if steps:
                blocks.append((label, make_pipeline(*steps), columns))
                stateful = True
            else:
                # Passthrough, or ColumnTransformer would drop these columns.
                blocks.append((label, "passthrough", columns))

        if not stateful:
            return None
        return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)

    def _roles(self):
        """The (columns, transform spec, label) triple for each declared column role."""
        return (
            (self.continuous, self.continuous_transform, "continuous"),
            (self.categorical, self.categorical_transform, "categorical"),
        )

    @staticmethod
    def _resolve(spec: Any) -> list:
        """Normalise a transform spec to a list of unfitted transformer instances."""
        if spec is None:
            return []
        if isinstance(spec, str):
            raise ValueError(
                f"Transforms must be scikit-learn transformer instances, not the string "
                f"'{spec}'. Pass e.g. SimpleImputer(strategy='median'), a list of them, or "
                f"None to pass the columns through untouched. There is no shorthand: what "
                f"you preprocess with is a modelling decision and belongs in your script."
            )
        return list(spec) if isinstance(spec, list | tuple) else [spec]

    # ------------------------------------------------------------------ #
    # fold discipline
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        return self._preprocessor_spec is None or self._preprocessor is not None

    def subset(self, indices: Sequence[int]) -> TabularDataset:
        """Return a new dataset over ``indices``, slicing the frame in step."""
        clone = super().subset(indices)
        clone._frame = self._frame.iloc[list(indices)].reset_index(drop=True)
        clone._matrix = None if self._matrix is None else self._matrix[list(indices)]
        return clone

    def fit_on(self, indices: Sequence[int]) -> TabularDataset:
        """Fit transforms on ``indices`` and return a new dataset over them.

        Args:
            indices (Sequence[int]): Positional indices of the training rows.

        Returns:
            TabularDataset: A new, fitted instance. ``self`` is untouched, so folds
            stay independent and can be fitted in parallel.
        """
        clone = self.subset(indices)
        if self._preprocessor_spec is not None:
            clone._preprocessor = sk_clone(self._preprocessor_spec)
            clone._preprocessor.fit(clone._frame[self.feature_columns])
        clone._materialise()
        return clone

    def transform(self, other: TabularDataset) -> TabularDataset:
        """Apply this dataset's fitted transforms to ``other``'s rows.

        Args:
            other (TabularDataset): Held-out data to prepare.

        Returns:
            TabularDataset: A new instance over ``other``'s rows, using ``self``'s
            fitted statistics.

        Raises:
            NotFittedError: If ``self`` has unfitted transforms.
        """
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} has unfitted transforms; call fit_on() first.")
        clone = copy.copy(other)
        clone._preprocessor = self._preprocessor
        clone._materialise()
        return clone

    def _materialise(self) -> None:
        """Compute the feature matrix for this dataset's rows.

        Done once per fitted instance rather than per item: the table is small, and
        calling scikit-learn's ``transform`` inside ``__getitem__`` would dominate
        the training loop.
        """
        frame = self._frame[self.feature_columns]
        if self._preprocessor is None:
            values = frame.to_numpy(dtype="float64")
            names = list(self.feature_columns)
        else:
            values = self._preprocessor.transform(frame)
            names = list(self._preprocessor.get_feature_names_out())
        # np.array (not asarray) forces a writable copy as torch warns on read-only views.
        self._matrix = torch.as_tensor(np.array(values, dtype="float64"), dtype=torch.float32)
        self.feature_names = names

    #: Suggested fix per role, quoted back at the user in the messages below.
    _ENCODE_HINT = "categorical_transform=OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
    _IMPUTE_HINT = {
        "continuous": "continuous_transform=SimpleImputer(strategy='median')",
        "categorical": "categorical_transform=SimpleImputer(strategy='most_frequent')",
    }

    def _check_untransformed_roles(self) -> None:
        """Refuse to emit a feature matrix that no transform was declared to clean up.

        Checked **per role**, at construction, because declaring a transform for one
        role does nothing for the other. Checking the feature block as a whole would
        let a half-declared spec through -- scaled continuous columns alongside raw
        string categoricals -- which then dies much later, inside the numeric cast,
        as ``could not convert string to float: 'F'``.

        Both failures are silent or opaque otherwise: non-numeric columns reach that
        cast, and missing values sail through as NaN into the model.
        """
        for columns, spec, label in self._roles():
            if not columns or self._resolve(spec):
                continue  # nothing declared for this role is the only case to police
            frame = self._frame[columns]

            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
            if non_numeric:
                fix = (
                    f"Add e.g. {self._ENCODE_HINT}."
                    if label == "categorical"
                    else f"Declare them as categorical instead, with {self._ENCODE_HINT}."
                )
                raise ValueError(
                    f"{label.capitalize()} column(s) {non_numeric} are not numeric and no "
                    f"{label}_transform was declared for them. {fix}"
                )

            missing = frame.isna().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                raise ValueError(
                    f"Missing values in {missing.to_dict()} and no {label}_transform was "
                    f"declared to handle them. Add e.g. {self._IMPUTE_HINT[label]}, or resolve "
                    f"them upstream. Passing NaN into a model silently is never right."
                )

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #

    def get_by_id(self, identifier) -> Tensor:
        """Return the ``(n_features,)`` feature vector for one sample.

        Raises:
            NotFittedError: If transforms are declared but not yet fitted.
        """
        if self._matrix is None:
            raise NotFittedError(
                f"{type(self).__name__} has {len(self._pending())} unfitted transform(s) "
                f"({', '.join(self._pending())}). Use cross_validate(), or fit_on(indices) "
                f"to fit explicitly. For untransformed values, use .frame"
            )
        return self._matrix[self._row_of[identifier]]

    @property
    def frame(self) -> pd.DataFrame:
        """The untransformed table for this dataset's rows. The exploration hatch."""
        return self._frame

    @property
    def feature_columns(self) -> list[str]:
        """Declared feature columns, continuous first."""
        return self.continuous + self.categorical

    @property
    def n_features(self) -> int:
        """Feature count after encoding. Equals the declared column count until fitted."""
        return len(self.feature_names) if self.feature_names else len(self.feature_columns)

    def split(self, test_size: float = 0.2, random_state: int | None = None):
        """Split into two datasets, stratified on event status when a target is present.

        Stratifying matters at this cohort size: an unstratified 20% split of a few
        hundred patients can easily land with a badly skewed event rate.

        Args:
            test_size (float, optional): Proportion held out. Defaults to 0.2.
            random_state (int | None, optional): Seed.

        Returns:
            tuple[TabularDataset, TabularDataset]: Train and test, both unfitted.

        Raises:
            RuntimeError: If called on a fitted dataset. Both halves would inherit
                statistics fitted across all rows, so the held-out half would be
                standardised using its own data -- silently, with no error.
        """
        if self._preprocessor is not None:
            raise RuntimeError(
                "split() on a fitted dataset would give both halves transforms fitted "
                "across all rows, leaking the held-out half into its own preprocessing. "
                "Split first, then fit_on() the training half and transform() the rest."
            )
        stratify = None if self.target is None else self.target.events_for(self.identifiers)
        train_idx, test_idx = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=random_state, stratify=stratify
        )
        return self.subset(sorted(train_idx)), self.subset(sorted(test_idx))

    def describe_transforms(self) -> str:
        """Human-readable summary of the resolved transforms and their fitted status."""
        lines = []
        for columns, spec, label in self._roles():
            if not columns:
                continue
            steps = self._resolve(spec)
            rendered = " -> ".join(type(s).__name__ for s in steps) if steps else "passthrough"
            lines.append(f"{label:<12}({len(columns)} cols): {rendered}")
        status = (
            "fitted"
            if self._preprocessor is not None
            else ("none declared" if self._preprocessor_spec is None else "unfitted - fitted per fold")
        )
        lines.append(f"{'status':<12}: {status}")
        return "\n".join(lines)

    def _pending(self) -> list[str]:
        """Names of transformer classes that are declared but not yet fitted."""
        if self._preprocessor_spec is None or self._preprocessor is not None:
            return []
        names: list[str] = []
        for _, pipeline, _ in self._preprocessor_spec.transformers:
            if pipeline == "passthrough":
                continue
            names.extend(type(step).__name__ for _, step in pipeline.steps)
        return names

    def __repr__(self) -> str:
        parts = [f"{len(self)} samples", f"{len(self.feature_columns)} columns"]
        if self.feature_names:
            parts.append(f"{len(self.feature_names)} features")
        if self.target is not None:
            parts.append(self.target.summarise(self.identifiers))
        parts.append("fitted" if self._preprocessor is not None else "unfitted")
        return f"{type(self).__name__}({' | '.join(parts)})"
