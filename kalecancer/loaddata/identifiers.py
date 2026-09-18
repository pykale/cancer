"""Patient identifier checks shared by modalities, targets and datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


class _NotFoundError(KeyError):
    """A KeyError whose message prints as written (KeyError would quote and escape it)."""

    def __str__(self) -> str:
        return str(self.args[0])


def _as_ids(values: Iterable, what: str) -> pd.Index:
    index = pd.Index(list(values))
    not_str = [i for i in index if not isinstance(i, str)]
    if not_str:
        raise TypeError(
            f"{what}: patient ids must be str, got {type(not_str[0]).__name__} {not_str[0]!r}. "
            "pd.read_json turns '001' into 1 unless dtype={'patient_id': str}"
        )
    if not index.is_unique:
        raise ValueError(f"{what}: duplicate ids {index[index.duplicated()].unique()[:5].tolist()}")
    return index


def _check_leading_zeros(sources: Mapping[str, Iterable[str]]) -> None:
    first_seen: dict[str, tuple[str, str]] = {}
    for source, ids in sources.items():
        for pid in ids:
            other, other_source = first_seen.setdefault(pid.lstrip("0") or "0", (pid, source))
            if other != pid and other_source != source:
                raise ValueError(
                    f"id {other!r} in {other_source} and {pid!r} in {source} differ only by leading zeros; "
                    "read ids as zero-padded strings everywhere"
                )
