"""Access to datasets published as remote ZIP archives.

Large imaging archives are usually distributed as one multi-gigabyte ZIP while an
experiment needs a fraction of the members. A ZIP keeps its index at the end, so a
server supporting HTTP range requests lets the index be read and individual members
extracted without transferring the archive.

A dataset is declared by subclassing :class:`ArchiveDataset` with its archives; the
fetching, caching and member selection are inherited:

    >>> class MyDataset(ArchiveDataset):
    ...     name = "mine"
    ...     archives = {"images": RemoteArchive("https://host/images.zip")}
    >>> paths = MyDataset().fetch_matching("images", suffix=".h5", limit=10)

Dataset-specific layout -- which archive holds what, how members are named -- belongs
in the subclass, which is normally an experiment concern and so lives with the
example rather than here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from kalecancer.loaddata.sources import extract_members, open_remote_zip

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "kalecancer"


class DatasetAccessError(RuntimeError):
    """Raised when a dataset cannot be prepared."""


@dataclass(frozen=True)
class RemoteArchive:
    """A downloadable archive belonging to a dataset."""

    url: str
    description: str = ""


def select_groups(members: Sequence[str], limit: int, pattern: re.Pattern[str]) -> list[str]:
    """Keep the members of the first ``limit`` groups, by sorted group identifier.

    Selecting by identifier rather than by position makes a given limit always yield
    the same subset, and keeps every member of a chosen group together -- which is
    what stops a partial download from splitting one subject across the boundary.

    Args:
        members: Archive member names.
        limit: Number of groups to keep; 0 keeps all members.
        pattern: Applied to each member's stem, exposing a ``patient_id`` named group.
    """
    if not limit:
        return sorted(members)

    grouped: dict[str, list[str]] = {}
    for member in members:
        match = pattern.match(Path(member).stem)
        if match:
            grouped.setdefault(match.group("patient_id"), []).append(member)

    return [member for group in sorted(grouped)[:limit] for member in sorted(grouped[group])]


class ArchiveDataset:
    """A dataset distributed as remote ZIP archives, cached locally on first use.

    Attributes:
        name: Cache subdirectory, so several datasets can share one cache root.
        archives: Archives by key, declared by the subclass.
    """

    name: str = "dataset"
    archives: dict[str, RemoteArchive] = {}

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.root = Path(cache_dir or DEFAULT_CACHE_DIR) / self.name

    def fetch(self, archive: str, select: Callable[[list[str]], list[str]]) -> list[Path]:
        """Extract the members ``select`` chooses from the archive index.

        Args:
            archive: Key into :attr:`archives`.
            select: Given every member name, returns those to extract.

        Returns:
            Local paths of the extracted members, already-cached ones included.

        Raises:
            DatasetAccessError: If the archive is unknown or nothing was selected.
        """
        if archive not in self.archives:
            raise DatasetAccessError(f"unknown archive {archive!r}; available: {sorted(self.archives)}")

        source = self.archives[archive]
        handle_archive, handle = open_remote_zip(source.url)
        with handle_archive:
            selected = select(handle_archive.namelist())
            if not selected:
                raise DatasetAccessError(f"no matching members in {source.url}")
            paths = extract_members(handle_archive, selected, self.root)

        logger.info(
            "%s/%s: %d members, %.1f MB of the %.2f GB archive",
            self.name,
            archive,
            len(selected),
            handle.bytes_fetched / 1e6,
            handle.size / 1e9,
        )
        return paths

    def fetch_matching(
        self,
        archive: str,
        prefix: str = "",
        suffix: str = "",
        limit: int = 0,
        group_pattern: re.Pattern[str] | None = None,
    ) -> list[Path]:
        """Extract members by name, optionally keeping only the first ``limit`` groups.

        Args:
            archive: Key into :attr:`archives`.
            prefix: Keep members whose name starts with this.
            suffix: Keep members whose name ends with this.
            limit: Number of groups to keep; 0 keeps every match.
            group_pattern: Required when ``limit`` is set, see :func:`select_groups`.

        Raises:
            DatasetAccessError: If ``limit`` is set without ``group_pattern``.
        """
        if limit and group_pattern is None:
            raise DatasetAccessError("group_pattern is required to limit the number of groups")

        def select(members: list[str]) -> list[str]:
            matched = [name for name in members if name.startswith(prefix) and name.endswith(suffix)]
            if limit:
                assert group_pattern is not None
                return select_groups(matched, limit, group_pattern)
            return sorted(matched)

        return self.fetch(archive, select)

    def fetch_named(self, archive: str, filename: str) -> Path:
        """Extract the first member with the given file name.

        Raises:
            DatasetAccessError: If no member has that name.
        """
        return self.fetch(archive, lambda members: [n for n in members if Path(n).name == filename][:1])[0]


def resolve_paths(cfg, fetch: Callable[[], tuple[Path, Path]] | None = None) -> tuple[Path, Path]:
    """Resolve ``DATASET.SOURCE`` to a feature root and a clinical file.

    ``"local"`` uses the configured paths; any other source calls ``fetch``, which the
    caller supplies for the dataset in question. Both return local paths, so nothing
    downstream depends on where the data came from.

    Raises:
        DatasetAccessError: If a remote source is configured without a fetcher.
    """
    if cfg.DATASET.SOURCE == "local":
        return Path(cfg.DATASET.FEATURE_ROOT), Path(cfg.DATASET.CLINICAL_PATH)
    if fetch is None:
        raise DatasetAccessError(
            f"DATASET.SOURCE is {cfg.DATASET.SOURCE!r} but no fetcher was supplied; "
            "pass one, or use 'local' with DATASET.FEATURE_ROOT and DATASET.CLINICAL_PATH"
        )
    return fetch()
