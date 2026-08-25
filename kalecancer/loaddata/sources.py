"""Random access to remote ZIP archives over HTTP.

Published imaging archives are often distributed as a single large ZIP, while an
experiment usually needs a handful of members. A ZIP stores its index at the end of
the file, so a server supporting HTTP range requests allows the index to be read and
individual members extracted without transferring the archive.
"""

from __future__ import annotations

import io
import logging
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

logger = logging.getLogger(__name__)

READ_BUFFER = 1024 * 1024
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


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
