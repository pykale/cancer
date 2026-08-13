from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from kalecancer.loaddata.dataset import BaseDataset

ColumnSpec = str | int | Sequence[str | int]


class TabularDataset(BaseDataset):
    """Tabular dataset handler for csv and JSON file formats."""

    # A table always has a file (unlike BaseDataset).
    path: Path
    _table: pd.DataFrame

    def __init__(
        self,
        path: str | Path,
        identifier_column: ColumnSpec,
        target_column: ColumnSpec | None = None,
        feature_columns: ColumnSpec | None = None,
    ):
        """Describe a table by naming the role each of its columns plays.

        Every column argument takes a single label or a sequence of them.

        Args:
            path (str | Path): Relative path to tabular data file.
            identifier_column (str | int | Sequence[str | int]):
                Column(s) containing the identifier(s). Several columns form a composite key.
            target_column (str | int | Sequence[str | int] | None, optional):
                Column(s) containing the target variable(s). If None provided, we assume no target variable is
                available (unsupervised). Defaults to None.
            feature_columns (str | int | Sequence[str | int] | None, optional):
                Column(s) containing the feature(s). If none provided, all columns except target and identifier will be
                used, in the order they appear in the file. Defaults to None.

        Raises:
            ValueError: If a column argument is malformed, no identifier is named, or one column is
                claimed for two roles at once.
        """
        super().__init__(path)

        self.identifier_column = _as_column_list(identifier_column, "identifier_column")
        if not self.identifier_column:
            raise ValueError("identifier_column must name at least one column.")
        self.target_column = _as_column_list(target_column, "target_column")

        self._requested_features = _as_column_list(feature_columns, "feature_columns")
        self.feature_columns: list[str | int] = list(self._requested_features)

        targets = set(self.target_column)
        leaked = [column for column in self.feature_columns if column in targets]
        if leaked:
            raise ValueError(
                f"Column(s) {leaked!r} are listed as both feature and target, which would feed the target back in as "
                f"model input."
            )

        claimed = targets.union(self.feature_columns)
        misused_identifiers = [column for column in self.identifier_column if column in claimed]
        if misused_identifiers:
            raise ValueError(
                f"Identifier column(s) {misused_identifiers!r} are also listed as a feature or target. Identifiers "
                f"become the index and are not carried as data."
            )

    def _require_columns(self, table: pd.DataFrame, columns: list[str | int], role: str) -> None:
        """Raise if any of ``columns`` is missing, naming what the file does hold."""
        missing = [column for column in columns if column not in table.columns]
        if missing:
            raise ValueError(
                f"{self.path} has no {role} column(s) {missing!r}. Available columns: {list(table.columns)!r}"
            )

    def _loader(self) -> None:
        """Read the file, check the named columns exist, and index the table by identifier."""
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            table = pd.read_csv(self.path)
        elif suffix == ".json":
            table = pd.read_json(self.path)
        else:
            raise ValueError(f"Unsupported file type {self.path.suffix!r} for tabular dataset.")

        self._require_columns(table, self.identifier_column, "identifier")
        if table[self.identifier_column].duplicated().any():
            raise ValueError(
                f"{self.path} must hold one row per identifier but {self.identifier_column!r} has duplicates."
            )

        table = table.set_index(self.identifier_column, drop=True)
        self._require_columns(table, self.target_column, "target")

        if self._requested_features:
            self._require_columns(table, self._requested_features, "feature")
            self.feature_columns = list(self._requested_features)
        else:
            targets = set(self.target_column)
            self.feature_columns = [column for column in table.columns if column not in targets]

        self._table = table[self.feature_columns + self.target_column]
        self.identifiers = list(self._table.index)

    def get_by_id(self, identifier) -> tuple[pd.Series, pd.Series] | pd.Series:
        """Return one row's features, paired with its target(s) when the dataset has any."""
        self.load_data()

        row = self._table.loc[identifier]
        features = row[self.feature_columns]
        if not self.target_column:
            return features

        return features, row[self.target_column]


def _as_column_list(spec: ColumnSpec | None, argument: str) -> list[str | int]:
    """Normalise a column argument to a list of labels."""
    if spec is None:
        return []
    if isinstance(spec, (str | int)):
        return [spec]
    if not isinstance(spec, Sequence):
        raise ValueError(f"{argument} must be a column label or a sequence of labels, got {type(spec).__name__}.")

    columns = list(spec)
    invalid = [column for column in columns if not isinstance(column, (str | int))]
    if invalid:
        raise ValueError(f"{argument} must contain only str or int labels, got {invalid!r}.")
    return columns
