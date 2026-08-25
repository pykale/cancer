"""Tests for remote archive access, served from a local HTTP server."""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import zipfile
from pathlib import Path

import pytest

from kalecancer.loaddata.sources import HttpRangeFile, RemoteArchiveError, extract_members, open_remote_zip

MEMBERS = {"data/a.txt": b"alpha" * 200, "data/b.txt": b"beta" * 300, "data/c.bin": b"\x00\x01" * 500}


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves one directory with byte-range support, as a real archive host does."""

    directory: Path

    def log_message(self, *args) -> None:  # noqa: D102 - keep the test output clean
        pass

    def _payload(self) -> bytes | None:
        target = self.directory / self.path.lstrip("/")
        return target.read_bytes() if target.is_file() else None

    def do_HEAD(self) -> None:  # noqa: N802
        payload = self._payload()
        if payload is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        payload = self._payload()
        if payload is None:
            self.send_error(404)
            return

        requested = self.headers.get("Range")
        if requested:
            start, end = requested.replace("bytes=", "").split("-")
            chunk = payload[int(start) : int(end) + 1]
            self.send_response(206)
        else:
            chunk = payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(chunk)


def _serve(root: Path, handler_class) -> tuple[str, socketserver.TCPServer]:
    handler = (
        type("Bound", (handler_class,), {"directory": root})
        if handler_class is _RangeHandler
        else functools.partial(handler_class, directory=str(root))
    )
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


@pytest.fixture(scope="module")
def archive_server(tmp_path_factory) -> str:
    """Serve a range-capable host holding a ZIP, and return the archive URL."""
    root = tmp_path_factory.mktemp("served")
    with zipfile.ZipFile(root / "archive.zip", "w") as archive:
        for name, payload in MEMBERS.items():
            archive.writestr(name, payload)
    (root / "plain.bin").write_bytes(b"not a zip at all" * 100)

    base, server = _serve(root, _RangeHandler)
    yield f"{base}/archive.zip"
    server.shutdown()


@pytest.fixture(scope="module")
def no_range_server(tmp_path_factory) -> str:
    """Python's stock handler advertises no range support."""
    root = tmp_path_factory.mktemp("norange")
    (root / "archive.zip").write_bytes(b"PK\x03\x04padding")

    base, server = _serve(root, http.server.SimpleHTTPRequestHandler)
    yield f"{base}/archive.zip"
    server.shutdown()


def test_reports_the_archive_size(archive_server: str) -> None:
    handle = HttpRangeFile(archive_server)

    assert handle.size > 0


def test_reads_a_byte_range(archive_server: str) -> None:
    handle = HttpRangeFile(archive_server)

    assert handle.read(2) == b"PK"


def test_supports_seeking(archive_server: str) -> None:
    handle = HttpRangeFile(archive_server)

    handle.seek(0, 2)
    assert handle.tell() == handle.size

    handle.seek(-2, 2)
    assert len(handle.read(2)) == 2


def test_unreachable_url_is_reported() -> None:
    with pytest.raises(RemoteArchiveError, match="cannot reach"):
        HttpRangeFile("http://127.0.0.1:1/missing.zip", timeout=5)


def test_server_without_range_support_is_reported(no_range_server: str) -> None:
    with pytest.raises(RemoteArchiveError, match="does not support HTTP range requests"):
        HttpRangeFile(no_range_server)


def test_lists_members_without_downloading_the_archive(archive_server: str) -> None:
    """Only the archive index is transferred, not the member payloads."""
    archive, handle = open_remote_zip(archive_server)

    with archive:
        assert set(archive.namelist()) == set(MEMBERS)
        assert handle.bytes_fetched < handle.size


def test_extracts_only_the_requested_members(archive_server: str, tmp_path: Path) -> None:
    archive, _ = open_remote_zip(archive_server)

    with archive:
        paths = extract_members(archive, ["data/a.txt"], tmp_path)

    assert [p.name for p in paths] == ["a.txt"]
    assert paths[0].read_bytes() == MEMBERS["data/a.txt"]
    assert not (tmp_path / "data" / "b.txt").exists()


def test_extraction_preserves_member_paths(archive_server: str, tmp_path: Path) -> None:
    archive, _ = open_remote_zip(archive_server)

    with archive:
        paths = extract_members(archive, list(MEMBERS), tmp_path)

    assert {p.relative_to(tmp_path).as_posix() for p in paths} == set(MEMBERS)


def test_existing_files_are_not_refetched(archive_server: str, tmp_path: Path) -> None:
    archive, handle = open_remote_zip(archive_server)

    with archive:
        extract_members(archive, ["data/a.txt"], tmp_path)
        after_first = handle.bytes_fetched
        extract_members(archive, ["data/a.txt"], tmp_path)

    assert handle.bytes_fetched == after_first


def test_refetching_can_be_forced(archive_server: str, tmp_path: Path) -> None:
    archive, _ = open_remote_zip(archive_server)
    target = tmp_path / "data" / "a.txt"

    with archive:
        extract_members(archive, ["data/a.txt"], tmp_path)
        target.write_bytes(b"corrupted")
        extract_members(archive, ["data/a.txt"], tmp_path, skip_existing=False)

    assert target.read_bytes() == MEMBERS["data/a.txt"]


def test_a_cached_file_is_left_untouched(archive_server: str, tmp_path: Path) -> None:
    """Caching is by presence, so an existing file is not rewritten."""
    archive, _ = open_remote_zip(archive_server)
    target = tmp_path / "data" / "a.txt"

    with archive:
        extract_members(archive, ["data/a.txt"], tmp_path)
        target.write_bytes(b"local edit")
        extract_members(archive, ["data/a.txt"], tmp_path)

    assert target.read_bytes() == b"local edit"


def test_a_url_that_is_not_a_zip_is_reported(archive_server: str) -> None:
    not_an_archive = archive_server.replace("archive.zip", "plain.bin")

    with pytest.raises(RemoteArchiveError, match="not a readable ZIP archive"):
        open_remote_zip(not_an_archive)


def test_a_member_escaping_the_destination_is_refused(tmp_path: Path) -> None:
    archive_path = tmp_path / "escaping.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "should never be written")

    with zipfile.ZipFile(archive_path) as archive, pytest.raises(RemoteArchiveError, match="outside"):
        extract_members(archive, ["../outside.txt"], tmp_path / "cache")

    assert not (tmp_path / "outside.txt").exists()


def test_a_directory_member_becomes_a_directory(tmp_path: Path) -> None:
    archive_path = tmp_path / "with_dir.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/", "")

    with zipfile.ZipFile(archive_path) as archive:
        extracted = extract_members(archive, ["folder/"], tmp_path / "cache")

    assert extracted[0].is_dir()


def test_an_absent_member_is_named_in_the_error(tmp_path: Path) -> None:
    archive_path = tmp_path / "small.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("present.txt", "here")

    with zipfile.ZipFile(archive_path) as archive, pytest.raises(RemoteArchiveError, match="absent.txt"):
        extract_members(archive, ["absent.txt"], tmp_path / "cache")
