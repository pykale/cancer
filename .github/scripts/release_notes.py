#!/usr/bin/env python3
"""Write the CHANGELOG.md section for a version, to use as its GitHub release notes.

Usage: release_notes.py <version> <changelog-file> <output-file>

The section is everything below the "# Version  <version>" header up to the next "# Version" header. The release
workflow fails when the section is missing, so a release is never published without the changelog entry that
changelog.yml writes to the release pull request.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECTION_RE = re.compile(r"^#\s+Version\s+(?P<version>\S+)\s*$")


def normalise_version(version: str) -> str:
    """Normalises a version string so tag names and changelog headers compare equal.

    Args:
        version (str): A version as written in a tag, title or header, e.g. "v0.3.0" or " 0.3.0RC1".

    Returns:
        str: The version without surrounding whitespace or a leading "v", in lower case, e.g. "0.3.0rc1".
    """
    return version.strip().lstrip("vV").lower()


def extract_section(changelog: str, version: str) -> str | None:
    """Returns the body of a version's section in the changelog.

    Args:
        changelog (str): The content of the changelog file.
        version (str): The version to look for, with or without a leading "v".

    Returns:
        str | None: The section body without its header or surrounding blank lines, or None if there is no section
            for the version.
    """
    wanted = normalise_version(version)
    body: list[str] | None = None
    for line in changelog.splitlines():
        match = SECTION_RE.match(line)
        if match:
            if body is not None:
                break
            if normalise_version(match.group("version")) == wanted:
                body = []
            continue
        if body is not None:
            body.append(line)
    return None if body is None else "\n".join(body).strip("\n")


def main() -> int:
    """Runs the command-line interface: writes the release notes for a version.

    Returns:
        int: The exit code, 0 on success or 1 if the changelog has no section for the version.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="release version, e.g. 0.3.0 or v0.3.0rc1")
    parser.add_argument("changelog", type=Path, help="changelog file to read")
    parser.add_argument("output", type=Path, help="file to write the release notes to")
    args = parser.parse_args()

    changelog = args.changelog.read_text(encoding="utf-8") if args.changelog.is_file() else ""
    section = extract_section(changelog, args.version)
    if not section:
        sys.stdout.write(
            f"::error::{args.changelog.as_posix()} has no section for version {normalise_version(args.version)}. "
            "The changelog workflow writes it when the release pull request is opened.\n"
        )
        return 1
    args.output.write_text(section + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote release notes for version {normalise_version(args.version)} to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
