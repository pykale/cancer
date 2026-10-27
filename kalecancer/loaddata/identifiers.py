"""Patient identifier checks shared by modalities, targets and datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


class _NotFoundError(KeyError):
    """A KeyError whose message prints as written (KeyError would quote and escape it)."""

    def __str__(self) -> str:
        return str(self.args[0])


def _as_ids(values: Iterable, what: str) -> pd.Index:
    """``values`` as an Index of unique str named ``"id"``, in the order given (ADR 0003)."""
    values = list(values)
    not_str = [i for i in values if not isinstance(i, str)]
    if not_str:
        raise TypeError(
            f"{what}: patient ids must be str, got {type(not_str[0]).__name__} {not_str[0]!r}. "
            "pd.read_json turns '001' into 1 unless dtype={'patient_id': str}"
        )
    # Built after the check, because dtype="str" would silently turn 1 into "1"
    index = pd.Index(values, dtype="str", name="id")
    if not index.is_unique:
        raise ValueError(f"{what}: duplicate ids {index[index.duplicated()].unique()[:5].tolist()}")
    return index


def _requested_ids(ids: Iterable[str] | str, known: pd.Index | None, what: str) -> pd.Index:
    """The ids given to ``[]`` as an Index, in the order given. A single str gives a one-id Index.

    A key that is not ids, such as a position or a set, raises ``TypeError``. When ``known`` is given, ids it lacks
    raise a KeyError.
    """
    if isinstance(ids, str):
        ids = [ids]
    elif isinstance(ids, set | frozenset | pd.DataFrame) or not isinstance(ids, Iterable):
        raise TypeError(
            f"{what} is indexed by a str, or an ordered list of str, got {type(ids).__name__}. "
            "For the patient at position i, index with .ids[i]"
        )
    requested = _as_ids(ids, what)
    if len(requested) == 0:
        raise ValueError(f"{what}: no ids requested")
    if known is not None and (missing := [pid for pid in requested if pid not in known]):
        raise _NotFoundError(f"{what}: {len(missing)} unknown ids, e.g. {missing[:5]}")
    return requested


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
