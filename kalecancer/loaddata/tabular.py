"""Covariates read from a flat table, keyed by a sample identifier."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone as sk_clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from torch import Tensor

from kalecancer.loaddata.base import Cohort, Indices, NotFittedError
from kalecancer.loaddata.protocols import Target
from kalecancer.prepdata.tabular import TabularPreprocessor

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

    Identifiers are forced to string because pandas reads ``"001"`` as ``1``, which a
    slide manifest keyed on ``"001"`` then silently fails to match. This can only
    *keep* padding, never recover it -- a frame read without ``dtype={id: str}`` has
    already lost it.
    """
    frame = frame.reset_index(drop=True)
    if id_column in frame.columns:
        frame[id_column] = frame[id_column].astype(str)
    return frame


class TabularCohort(Cohort):
    """Covariates from a flat table, keyed by a sample identifier.

    Column roles and their preprocessing are declared with plain scikit-learn
    transformers. Nothing is fitted here and this object is never mutated:
    :meth:`fit_preprocessor` fits on the rows you name and hands back a separate
    artifact, which :meth:`~kalecancer.loaddata.base.Cohort.view` pairs with a row
    subset to make a dataset.

    Args:
        source (str | Path | pd.DataFrame): A ``.csv``, ``.tsv`` or ``.json`` file, or
            a loaded ``DataFrame`` -- the route for anything else pandas can read.
            Read it with ``dtype={identifier: str}`` or zero-padded ids are already
            lost. The frame is copied.
        identifier (str): Column holding sample identifiers. Must be unique.
        target (Target | None, optional): ``None`` for a pure feature provider used as
            a component of a larger cohort.
        continuous (Sequence[str], optional): Numeric feature columns.
        categorical (Sequence[str], optional): Categorical feature columns.
        continuous_transform (optional): A scikit-learn transformer, or a list applied
            in order. ``None`` passes the columns through. No default pipeline:
            imputation and scaling are modelling decisions and belong in your script.
        categorical_transform (optional): As above, for the categorical columns.
        name (str, optional): Key for this modality in ``PatientSample.modalities``.
            Defaults to ``"clinical"``.

    Example:
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
        >>> cohort = TabularCohort(
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
        >>> train_idx, test_idx = cohort.split(test_size=0.2, random_state=0, stratify=True)
        >>> prep = cohort.fit_preprocessor(train_idx)     # fitted on train rows only
        >>> train, test = cohort.view(train_idx, prep), cohort.view(test_idx, prep)

        A frame is the seam for anything between reading the file and binding the
        target, so the decision stays visible in your script:

        >>> frame = pd.read_csv("data/my_cohort.csv", dtype={"patient_id": str})
        >>> frame = frame[frame.vital_status.notna()]   # 24 patients, status not recorded
        >>> cohort = TabularCohort(frame, identifier="patient_id", target=target)
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        identifier: str,
        target: Target | None = None,
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

        # Seeded from a caller's frame, or overwritten by _load_index from self.path.
        # The empty placeholder is never read.
        self._frame: pd.DataFrame = source.copy() if isinstance(source, pd.DataFrame) else pd.DataFrame()
        self._spec: ColumnTransformer | None = None

        super().__init__(path=None if isinstance(source, pd.DataFrame) else source, name=name, target=target)

    def _has_index_source(self) -> bool:
        """Always: ``source`` is either a path to read or a frame already in hand."""
        return True

    # ------------------------------------------------------------------ #
    # index loading
    # ------------------------------------------------------------------ #

    def _load_index(self) -> None:
        """Read the table, validate the declared columns, bind the target.

        Reads everything, because a clinical table is small and index and payload are
        the same read here. WSI is what motivates keeping the phases apart.
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

        self._bind_target()
        self._spec = self._build_spec()
        self._check_untransformed_roles()

    def _bind_target(self) -> None:
        """Hand the target the columns it declared it needs, as arrays.

        Checked here rather than inside ``bind`` so a missing column fails at
        construction, named in the target's own vocabulary.
        """
        if self.target is None:
            return
        missing = [c for c in self.target.required_columns if c not in self._frame.columns]
        if missing:
            raise ValueError(
                f"{type(self.target).__name__} requires column(s) {missing}, which are not "
                f"in this table. Available: {sorted(self._frame.columns)}"
            )
        values = {c: self._frame[c].to_numpy() for c in self.target.required_columns}
        self.target.bind(self.identifiers, values)

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

    def _build_spec(self) -> ColumnTransformer | None:
        """Assemble an *unfitted* ColumnTransformer, or ``None`` if no role is stateful."""
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
    # fold-local fitting
    # ------------------------------------------------------------------ #

    def fit_preprocessor(self, indices: Indices) -> TabularPreprocessor:
        """Fit the declared transforms on ``indices`` only, and return the artifact.

        The cohort is not modified, so folds are independent and can be fitted in
        parallel.

        Returns:
            TabularPreprocessor: Fitted transforms, the resulting feature names, and
            the identifiers they were fitted on.
        """
        rows = list(indices)
        columns = self.feature_columns
        if self._spec is None:
            # Nothing stateful was declared. fitted_on stays empty: a passthrough
            # carries no row's information, so it cannot leak one.
            return TabularPreprocessor(None, columns, feature_names={self.name: list(columns)})

        transformer = sk_clone(self._spec)
        transformer.fit(self._frame.iloc[rows][columns])
        return TabularPreprocessor(
            transformer,
            columns,
            feature_names={self.name: list(transformer.get_feature_names_out())},
            fitted_on=frozenset(self.identifiers[i] for i in rows),
        )

    # ------------------------------------------------------------------ #
    # payload
    # ------------------------------------------------------------------ #

    def payload_bulk(self, identifiers, prep) -> dict[str, Tensor]:
        """Transform every row at once, which is what lets a view cache.

        Overridden because scikit-learn's per-call overhead would dominate the loop if
        ``transform`` ran once per row.
        """
        self._require(prep)
        rows = self.index_of(identifiers)
        return {self.name: prep.transform(self._frame.iloc[rows])}

    def payload(self, identifier: str, prep) -> dict[str, Tensor]:
        """Return the ``(n_features,)`` feature vector for one sample."""
        self._require(prep)
        row = self._row_of[identifier]
        return {self.name: prep.transform(self._frame.iloc[[row]])[0]}

    def _require(self, prep) -> None:
        """Refuse to serve values without a preprocessor.

        Refused even for a passthrough, so there is one rule rather than two: accepting
        ``None`` would let a cohort *with* declared transforms serve raw values.
        """
        if prep is None:
            raise NotFittedError(
                f"{type(self).__name__} was asked for values with no preprocessor.\n"
                f"  Fit one on this fold's training rows first:\n"
                f"      prep = cohort.fit_preprocessor(train_idx)\n"
                f"      train = cohort.view(train_idx, prep)\n"
                f"      val   = cohort.view(val_idx, prep)   # same prep, never refitted\n"
                f"  For the untransformed values, use cohort.frame."
            )

    #: Suggested fix per role, quoted back at the user in the messages below.
    _ENCODE_HINT = "categorical_transform=OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
    _IMPUTE_HINT = {
        "continuous": "continuous_transform=SimpleImputer(strategy='median')",
        "categorical": "categorical_transform=SimpleImputer(strategy='most_frequent')",
    }

    def _check_untransformed_roles(self) -> None:
        """Refuse to build a cohort whose features no transform was declared to clean up.

        Checked **per role**, because a transform on one role does nothing for the
        other. A half-declared spec -- scaled numbers beside raw strings -- would
        otherwise die much later as ``could not convert string to float: 'F'``, and
        missing values would sail through as NaN into the model.
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

    @property
    def frame(self) -> pd.DataFrame:
        """The untransformed table. The exploration hatch."""
        return self._frame

    @property
    def feature_columns(self) -> list[str]:
        """Declared feature columns, continuous first."""
        return self.continuous + self.categorical

    def describe_transforms(self) -> str:
        """Summary of the transforms this cohort *declares*.

        What a fold actually applied is its ``TabularPreprocessor``'s to answer.
        """
        lines = []
        for columns, spec, label in self._roles():
            if not columns:
                continue
            steps = self._resolve(spec)
            rendered = " -> ".join(type(s).__name__ for s in steps) if steps else "passthrough"
            lines.append(f"{label:<12}({len(columns)} cols): {rendered}")
        if not lines:
            return "no feature columns declared"
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [f"{len(self)} samples", f"{len(self.feature_columns)} columns"]
        if self.target is not None:
            summarise = getattr(self.target, "summarise", None)
            parts.append(summarise(self.identifiers) if summarise else type(self.target).__name__)
        return f"{type(self).__name__}({' | '.join(parts)})"
