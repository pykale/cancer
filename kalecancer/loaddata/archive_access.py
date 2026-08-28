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

import io
import logging
import re
import ssl
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import certifi

logger = logging.getLogger(__name__)

READ_BUFFER = 1024 * 1024
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "kalecancer"

# --------------------------------------------------------------------------- #
# Transport: reading a remote ZIP without downloading it
# --------------------------------------------------------------------------- #


class RemoteArchiveError(RuntimeError):
    """Raised when a remote archive cannot be read."""


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file backed by HTTP range requests.

    Args:
        url: Archive URL. The server must report ``Accept-Ranges: bytes``.
        timeout: Per-request timeout in seconds.

    Raises:
        RemoteArchiveError: If the URL is unreachable or does not support ranges.
    """

    def __init__(self, url: str, timeout: float = 120.0) -> None:
        self.url = url
        self.timeout = timeout
        self.position = 0
        self.request_count = 0
        self.bytes_fetched = 0

        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=timeout, context=_SSL_CONTEXT) as response:
                self.size = int(response.headers["Content-Length"])
                accepts_ranges = response.headers.get("Accept-Ranges") == "bytes"
        except Exception as error:  # noqa: BLE001 - surfaced with the URL for diagnosis
            raise RemoteArchiveError(f"cannot reach {url}: {error}") from error

        if not accepts_ranges:
            raise RemoteArchiveError(f"{url} does not support HTTP range requests, so it must be downloaded in full")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        else:
            self.position = self.size + offset
        return self.position

    def readinto(self, buffer) -> int:  # type: ignore[override]
        length = min(len(buffer), self.size - self.position)
        if length <= 0:
            return 0

        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{self.position + length - 1}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=_SSL_CONTEXT) as response:
                data = response.read()
        except Exception as error:  # noqa: BLE001 - surfaced with the byte range for diagnosis
            raise RemoteArchiveError(f"range request failed at byte {self.position} of {self.url}: {error}") from error

        buffer[: len(data)] = data
        self.request_count += 1
        self.bytes_fetched += len(data)
        self.position += len(data)
        return len(data)


def open_remote_zip(url: str, timeout: float = 120.0) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    """Open a remote ZIP for selective reading.

    Only the archive index is transferred, which is typically a fraction of a percent
    of the archive.

    Returns:
        The opened archive and the underlying file, whose ``bytes_fetched`` records
        how much has been transferred.

    Raises:
        RemoteArchiveError: If the archive cannot be read.
    """
    handle = HttpRangeFile(url, timeout=timeout)
    try:
        archive = zipfile.ZipFile(io.BufferedReader(handle, buffer_size=READ_BUFFER))
    except zipfile.BadZipFile as error:
        raise RemoteArchiveError(f"{url} is not a readable ZIP archive: {error}") from error
    return archive, handle


def _member_target(destination: Path, member: str) -> Path:
    """Resolve ``member`` inside ``destination``, refusing paths that escape it.

    Archive member names are attacker-controlled when the archive is remote, and a
    name such as ``../config`` would otherwise be written outside the cache.
    """
    root = destination.resolve()
    target = (root / member).resolve()
    if target != root and root not in target.parents:
        raise RemoteArchiveError(f"archive member {member!r} would be written outside {destination}")
    return target


def extract_members(
    archive: zipfile.ZipFile,
    members: list[str],
    destination: str | Path,
    skip_existing: bool = True,
) -> list[Path]:
    """Extract named members, preserving their paths inside ``destination``.

    Args:
        archive: An open archive, local or remote.
        members: Member names to extract.
        destination: Directory to write into.
        skip_existing: Leave already-extracted files untouched, making repeated calls
            inexpensive.

    Returns:
        The extracted paths, in the order requested.

    Raises:
        RemoteArchiveError: If a member is absent, or names a path outside
            ``destination``.
    """
    destination = Path(destination)
    paths = []
    for index, member in enumerate(members, start=1):
        try:
            info = archive.getinfo(member)
        except KeyError as error:
            raise RemoteArchiveError(f"archive has no member named {member!r}") from error

        target = _member_target(destination, member)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            paths.append(target)
            continue
        if skip_existing and target.exists():
            paths.append(target)
            continue

        # Written beside the target and renamed once complete, so an interrupted
        # transfer cannot leave a truncated file that the next run treats as cached.
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        try:
            with archive.open(member) as source, partial.open("wb") as output:
                while chunk := source.read(READ_BUFFER):
                    output.write(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)

        paths.append(target)
        logger.info("fetched %d/%d %s", index, len(members), member)
    return paths


# --------------------------------------------------------------------------- #
# The dataset facade
# --------------------------------------------------------------------------- #


class DatasetAccessError(RuntimeError):
    """Raised when a dataset cannot be prepared."""


@dataclass(frozen=True)
class RemoteArchive:
    """A downloadable archive belonging to a dataset."""

    url: str
    description: str = ""


def _select_groups(members: Sequence[str], limit: int, pattern: re.Pattern[str]) -> list[str]:
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
            group_pattern: Required when ``limit`` is set, see :func:`_select_groups`.

        Raises:
            DatasetAccessError: If ``limit`` is set without ``group_pattern``.
        """
        if limit and group_pattern is None:
            raise DatasetAccessError("group_pattern is required to limit the number of groups")

        def select(members: list[str]) -> list[str]:
            matched = [name for name in members if name.startswith(prefix) and name.endswith(suffix)]
            if limit:
                assert group_pattern is not None
                return _select_groups(matched, limit, group_pattern)
            return sorted(matched)

        return self.fetch(archive, select)

    def fetch_named(self, archive: str, filename: str) -> Path:
        """Extract the first member with the given file name.

        Raises:
            DatasetAccessError: If no member has that name.
        """
        return self.fetch(archive, lambda members: [n for n in members if Path(n).name == filename][:1])[0]
