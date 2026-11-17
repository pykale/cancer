"""Tests for the release changelog scripts, configuration and workflow."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
CONFIG = ROOT / ".github" / "changelog-config.json"
WORKFLOW = ROOT / ".github" / "workflows" / "changelog.yml"

# Older releases with Markdown that must survive untouched: a hard line break
# (two trailing spaces, written as "\x20\x20" so editors and the pre-commit
# trailing-whitespace hook don't strip them) and blank lines in a code block.
HISTORY = (
    "# Version  0.2.0\n"
    "\n"
    "#### New Features\n"
    "\n"
    "* [#477](https://github.com/pykale/pykale/pull/477): Add Integrated Gradients\x20\x20\n"
    "  with a hard line break\n"
    "\n"
    "```python\n"
    "x = 1\n"
    "\n"
    "\n"
    "y = 2\n"
    "```\n"
    "\n"
    "# Version  0.1.2\n"
    "\n"
    "#### Bug Fixes\n"
    "\n"
    "* [#389](https://github.com/pykale/pykale/pull/389): Fix something\n"
)

GENERATED = """#### New Features

* [#590](https://github.com/pykale/pykale/pull/590): Add X

#### Bug Fixes


#### Other Changes

* [#591](https://github.com/pykale/pykale/pull/591): Tidy Y
"""


def load(name):
    """Loads a script from .github/scripts as a module, since that folder is not a package.

    Args:
        name (str): The script's file name without the ".py" extension.

    Returns:
        module: The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_version = load("release_version")
update_changelog = load("update_changelog")


# changelog configuration and workflow


def test_operational_labels_are_ignored():
    """Tests that workflow-only labels do not appear in release notes."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert "action failed" in config["ignore_labels"]


def test_workflow_collects_pull_requests_via_commits():
    """Guards against silently truncating release notes with the date-search defaults."""
    assert "fetchViaCommits: true" in WORKFLOW.read_text(encoding="utf-8")


def test_workflow_uses_trusted_files_and_short_lived_token():
    """Guards the trust boundary between pull request content and authenticated steps."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("persist-credentials: false") == 2
    assert "ref: ${{ github.event.pull_request.base.sha }}" in workflow
    assert "path: trusted-base" in workflow
    assert 'cp trusted-base/.github/scripts/release_version.py "$RUNNER_TEMP/release_version.py"' in workflow
    assert 'cp trusted-base/.github/scripts/update_changelog.py "$RUNNER_TEMP/update_changelog.py"' in workflow
    assert 'cp trusted-base/.github/changelog-config.json "$RUNNER_TEMP/changelog-config.json"' in workflow
    assert 'python3 "$RUNNER_TEMP/release_version.py"' in workflow
    assert 'python3 "$RUNNER_TEMP/update_changelog.py"' in workflow
    assert "configuration: ${{ runner.temp }}/changelog-config.json" in workflow
    assert "secrets.CHANGELOG_TOKEN" not in workflow
    assert "git -c core.hooksPath=/dev/null commit" in workflow
    assert workflow.index("git add .github/CHANGELOG.md") < workflow.index(
        "git diff --cached --quiet -- .github/CHANGELOG.md"
    )


# release_version.py


@pytest.mark.parametrize(
    "title, package_version, expected",
    [
        ("Release 0.3.0", "0.3.0", "0.3.0"),
        ("Release-0.3.0", "0.3.0", "0.3.0"),
        ("Release:0.3.0", "0.3.0", "0.3.0"),
        ("release_0.3.0", "0.3.0", "0.3.0"),
        ("release/v0.3.0", "0.3.0", "0.3.0"),
        ("[Release]0.3.0", "0.3.0", "0.3.0"),
        ("Release-v0.3.0RC1", "0.3.0rc1", "0.3.0rc1"),
        ("Prepare the 0.3.0 release", "0.3.0", "0.3.0"),
        ("0.3.0-release", "0.3.0", "0.3.0"),
        ("0.3.0/release", "0.3.0", "0.3.0"),
        ("0.3.0_release", "0.3.0", "0.3.0"),
        ("V0.3.0:RELEASE", "0.3.0", "0.3.0"),
        ("0.3.0rc1-release", "0.3.0rc1", "0.3.0rc1"),
        ("0.3.0., release", "0.3.0", "0.3.0"),
        ("Prepare the (v0.3.0) release", "0.3.0", "0.3.0"),
        ("RELEASE v0.3.0.", "0.3.0", "0.3.0"),
        ("Release (0.3.0)", "0.3.0", "0.3.0"),
        ("Release 0.3.0a1", "0.3.0a1", "0.3.0a1"),
        ("Release 0.3.0b2", "0.3.0b2", "0.3.0b2"),
        ("Release 0.3.0RC1", "0.3.0rc1", "0.3.0rc1"),
    ],
)
def test_release_version_accepted(title, package_version, expected):
    """Tests that plain releases and a/b/rc pre-releases matching the package version are accepted."""
    assert release_version.find_release_version(title, package_version) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Add MOGONET example",
        "Fix prerelease check in 0.3.0",
        "Update released notebooks for 0.3.0",
        "Releases 0.3.0",
        "Fix release publishing workflow",
        "Release use ubuntu-latest",
        "Update release workflow to Python 3.12",
        "Fix release notes link for 0.2.0",
        "Pin torch 2.6.0 in release job",
        "Release notes for 0.2.0 and 0.3.0",
        "Fix release workflow for package0.3.0",
        "Fix release workflow for paqueté0.3.0",
    ],
)
def test_release_version_not_a_release(title):
    """Tests that titles without the word "release" or without a version are not treated as releases."""
    assert release_version.find_release_version(title, "0.3.0") is None


@pytest.mark.parametrize(
    "title, message",
    [
        ("Release 0.3.0-beta.1", "Unsupported version '0.3.0-beta.1'"),
        ("0.3.0-beta.1-release", "Unsupported version '0.3.0-beta.1'"),
        ("0.3_release", "Unsupported version '0.3'"),
        ("0.3.0.post1/release", "Unsupported version '0.3.0.post1'"),
        ("Release v.0.3.0", "Unsupported version 'v.0.3.0'"),
        ("Release-0.3.0-beta.1", "Unsupported version '0.3.0-beta.1'"),
        ("Release:0.3.0.post1", "Unsupported version '0.3.0.post1'"),
        ("Release-0.3", "Unsupported version '0.3'"),
        ("release_0.3", "Unsupported version '0.3'"),
        ("release_0.3.0-beta.1", "Unsupported version '0.3.0-beta.1'"),
        ("Release:0.3.1", "does not match kalecancer/__init__.py (0.3.0)"),
        ("Release 0.2.0,0.3.0", "exactly one version, found: 0.2.0 0.3.0"),
        ("Release 0.3.0.post1", "Unsupported version '0.3.0.post1'"),
        ("Release 0.3.0dev1", "Unsupported version '0.3.0dev1'"),
        ("Release 0.2.0 and 0.3.0", "exactly one version, found: 0.2.0 0.3.0"),
        ("Release 0.3.1", "does not match kalecancer/__init__.py (0.3.0)"),
        # A truncated version must be reported, not mistaken for a title that merely mentions
        # releases: that let a release pull request pass with no changelog generated.
        ("Release 0.3", "Unsupported version '0.3'"),
        ("Release v0.3", "Unsupported version 'v0.3'"),
        ("Release 0.3 and 0.4", "exactly one version, found: 0.3 0.4"),
    ],
)
def test_release_version_rejected(title, message):
    """Tests that unsupported, ambiguous and mismatched versions raise an error saying what to fix."""
    with pytest.raises(release_version.ReleaseTitleError, match=message.replace("(", r"\(").replace(")", r"\)")):
        release_version.find_release_version(title, "0.3.0")


def test_release_version_reads_package_version(tmp_path):
    """Tests reading ``__version__`` from ``__init__.py``, and the error when it is missing."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text('"""Docs."""\n__version__ = "0.3.0rc1"\n', encoding="utf-8")
    assert release_version.read_package_version(init_file) == "0.3.0rc1"

    init_file.write_text("VERSION = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        release_version.read_package_version(init_file)


@pytest.mark.parametrize("assignment", ["__version__='0.3.0'", '__version__ ="0.3.0"', "__version__\t=\t'0.3.0'"])
def test_package_version_assignment_whitespace(tmp_path, assignment):
    """Accepts spacing variants without importing package code."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text(assignment + "\n", encoding="utf-8")
    assert release_version.read_package_version(init_file) == "0.3.0"


@pytest.mark.parametrize("content", [None, "VERSION = '0.3.0'", "__version__ = '0.3.0\""])
def test_package_version_cli_error(tmp_path, monkeypatch, capsys, content):
    """Reports missing files and malformed assignments as actionable workflow errors."""
    init_file = tmp_path / "__init__.py"
    if content is not None:
        init_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["release_version.py", str(init_file)])
    monkeypatch.setenv("PR_TITLE", "Release 0.3.0")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert release_version.main() == 1
    message = capsys.readouterr().out
    assert message.startswith("::error::")
    assert str(init_file) in message
    assert not output.exists()


@pytest.mark.parametrize(
    "title, exit_code, output",
    [
        ("Release 0.3.0", 0, "version=0.3.0\n"),
        ("Release-0.3.0", 0, "version=0.3.0\n"),
        ("0.3.0-release", 0, "version=0.3.0\n"),
        ("0.3.0-beta.1-release", 1, ""),
        ("release_0.3.0", 0, "version=0.3.0\n"),
        ("release_0.3", 1, ""),
        ("Release v.0.3.0", 1, ""),
        ("Release-0.3", 1, ""),
        ("Add MOGONET example", 0, "version=\n"),
        ("Update release workflow to Python 3.12", 0, "version=\n"),
        ("Fix release notes link for 0.2.0", 0, "version=\n"),
        ("Pin torch 2.6.0 in release job", 0, "version=\n"),
        ("Release 0.3.1", 1, ""),
    ],
)
def test_release_version_main(tmp_path, monkeypatch, capsys, title, exit_code, output):
    """Tests the command-line interface's exit code, ``$GITHUB_OUTPUT`` and error annotation."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text('__version__ = "0.3.0"\n', encoding="utf-8")
    github_output = tmp_path / "github_output"
    github_output.touch()
    monkeypatch.setenv("PR_TITLE", title)
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr("sys.argv", ["release_version.py", str(init_file)])

    assert release_version.main() == exit_code
    assert github_output.read_text(encoding="utf-8") == output
    if exit_code:
        assert capsys.readouterr().out.startswith("::error::")


# update_changelog.py


def run_updater(monkeypatch, version, generated, changelog):
    """Runs update_changelog.py's command-line interface with the given arguments.

    Args:
        monkeypatch (pytest.MonkeyPatch): The fixture used to set ``sys.argv``.
        version (str): The release version argument.
        generated (Path): The file generated by the action, rewritten to hold the comment body.
        changelog (Path): The changelog file to update.

    Returns:
        int: The exit code returned by ``main()``.
    """
    monkeypatch.setattr("sys.argv", ["update_changelog.py", version, str(generated), str(changelog)])
    return update_changelog.main()


def test_drop_empty_groups():
    """Tests that group headings without entries are removed and the others kept."""
    body = update_changelog.drop_empty_groups(GENERATED)
    assert "#### Bug Fixes" not in body
    assert body.startswith("#### New Features")
    assert body.endswith("Tidy Y")


def test_insert_keeps_history_byte_for_byte(tmp_path, monkeypatch):
    """Tests that a new section is inserted at the top and older releases are left byte for byte."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())
    generated = tmp_path / "generated.md"
    generated.write_text(GENERATED, encoding="utf-8")

    assert run_updater(monkeypatch, "v0.3.0", generated, changelog) == 0

    content = changelog.read_bytes().decode()
    section, rest = content.split("\n\n# Version  0.2.0", 1)
    assert section.startswith("# Version  0.3.0\n\n#### New Features\n\n* [#590]")
    assert "#### Bug Fixes" not in section
    # Trailing spaces and blank lines in older releases are kept unchanged.
    assert "# Version  0.2.0" + rest == HISTORY
    # The comment body is the new section.
    assert generated.read_text(encoding="utf-8") == section + "\n"


def test_existing_section_is_never_overwritten(tmp_path, monkeypatch):
    """Tests that a curated section for the version is kept and the fresh list only goes to the comment."""
    curated = "# Version  0.3.0\n\n#### New Features\n\n* Reworded by a maintainer\n\n" + HISTORY
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(curated.encode())
    generated = tmp_path / "generated.md"
    generated.write_text(GENERATED, encoding="utf-8")

    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0

    assert changelog.read_bytes().decode() == curated
    comment = generated.read_text(encoding="utf-8")
    assert "already has a section for version 0.3.0" in comment
    assert "* [#590](https://github.com/pykale/pykale/pull/590): Add X" in comment


def test_all_groups_empty_writes_placeholder(tmp_path, monkeypatch):
    """Tests that a placeholder entry is written when every group is empty."""
    changelog = tmp_path / "CHANGELOG.md"
    generated = tmp_path / "generated.md"
    generated.write_text("#### Other Changes\n\n", encoding="utf-8")

    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0
    assert changelog.read_text(encoding="utf-8") == "# Version  0.3.0\n\n* No changelog entries for this release.\n"


def test_missing_generated_file_fails(tmp_path, monkeypatch):
    """Tests that a missing generated file fails without touching the changelog."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())

    assert run_updater(monkeypatch, "0.3.0", tmp_path / "missing.md", changelog) == 1
    assert changelog.read_bytes().decode() == HISTORY


# Retrying the workflow after the changelog commit was already pushed


def git(repo, *args):
    """Runs a git command in a repository.

    Args:
        repo (Path): The repository directory.
        *args (str): The git subcommand and its arguments.

    Returns:
        str: The command's standard output, stripped.
    """
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_rerun_from_branch_tip_needs_no_second_commit(tmp_path, monkeypatch):
    """Tests a rerun whose first attempt pushed the changelog but failed to comment.

    The workflow checks out the branch tip, so the rerun sees the pushed
    section, changes nothing (no second commit that would be rejected as
    non-fast-forward) and still writes the comment body.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "release-0.3.0")
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "core.autocrlf", "false")
    changelog = repo / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())
    git(repo, "add", "CHANGELOG.md")
    git(repo, "commit", "-q", "-m", "Bump version to 0.3.0")
    event_sha = git(repo, "rev-parse", "HEAD")
    generated = tmp_path / "generated.md"

    # First attempt: inserts the section and pushes it, then fails to comment.
    generated.write_text(GENERATED, encoding="utf-8")
    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0
    git(repo, "commit", "-q", "-am", "Update CHANGELOG.md for version 0.3.0")
    pushed_sha = git(repo, "rev-parse", "HEAD")
    assert pushed_sha != event_sha

    # Start at the original event commit to distinguish replaying head.sha
    # from checking out the branch tip containing the first attempt's commit.
    git(repo, "checkout", "-q", "--detach", event_sha)
    assert git(repo, "rev-parse", "HEAD") == event_sha
    assert changelog.read_bytes().decode() == HISTORY

    # Rerun: checks out the branch tip, which already holds the section.
    git(repo, "checkout", "-q", "release-0.3.0")
    assert git(repo, "rev-parse", "HEAD") == pushed_sha
    generated.write_text(GENERATED, encoding="utf-8")
    assert run_updater(monkeypatch, "0.3.0", generated, changelog) == 0

    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "rev-parse", "HEAD") == pushed_sha
    assert "* [#590](https://github.com/pykale/pykale/pull/590): Add X" in generated.read_text(encoding="utf-8")


# Formatting, as in the hand-edited PyKale 0.2.2 changelog

RELEASE_022_GENERATED = """#### New Features

* [#616](https://github.com/pykale/pykale/pull/616): feat: add reusable evaluation metrics
* [#607](https://github.com/pykale/pykale/pull/607): feat: add named-input multimodal fusion

#### Bug Fixes

* [#619](https://github.com/pykale/pykale/pull/619): Fix MNIST-M download
* [#612](https://github.com/pykale/pykale/pull/612): fix: download SVHN test data from verified UCSD mirror

#### Code Improvements

* [#615](https://github.com/pykale/pykale/pull/615): Update matplotlib version requirement
* [#614](https://github.com/pykale/pykale/pull/614): build(deps-dev): bump setuptools from 84.0.0 to 84.2.1

#### Other Changes

* [#602](https://github.com/pykale/pykale/pull/602): Release 0.2.1
"""

RELEASE_022_FORMATTED = """#### New Features

* [#607](https://github.com/pykale/pykale/pull/607): Add named-input multimodal fusion
* [#616](https://github.com/pykale/pykale/pull/616): Add reusable evaluation metrics

#### Bug Fixes

* [#612](https://github.com/pykale/pykale/pull/612): Download SVHN test data from verified UCSD mirror
* [#619](https://github.com/pykale/pykale/pull/619): Fix MNIST-M download

#### Code Improvements

* [#614](https://github.com/pykale/pykale/pull/614): Bump setuptools from 84.0.0 to 84.2.1
* [#615](https://github.com/pykale/pykale/pull/615): Update matplotlib version requirement"""

release_notes = load("release_notes")


def test_dependency_updates_are_code_improvements():
    """Tests that dependency bumps are grouped as code improvements, as maintainers moved them in PyKale 0.2.2."""
    groups = {group["title"]: group["labels"] for group in json.loads(CONFIG.read_text(encoding="utf-8"))["categories"]}

    assert "dependencies" in groups["#### Code Improvements"]
    assert "dependencies" not in groups["#### Bug Fixes"]


@pytest.mark.parametrize(
    "title, expected",
    [
        ("feat: add named-input multimodal fusion", "Add named-input multimodal fusion"),
        ("build(deps-dev): bump setuptools from 84.0.0 to 84.2.1", "Bump setuptools from 84.0.0 to 84.2.1"),
        ("fix!: drop  the old   API", "Drop the old API"),
        ("Docs(readme): explain installs", "Explain installs"),
        ("Fix MNIST-M download", "Fix MNIST-M download"),
        ("add PyKale core API", "Add PyKale core API"),
        ("featured examples", "Featured examples"),
    ],
)
def test_clean_title(title, expected):
    """Tests that Conventional Commits prefixes go, spacing is collapsed and only the first letter is capitalised."""
    assert update_changelog.clean_title(title) == expected


def test_format_entries_matches_hand_edited_release():
    """Tests that formatting reproduces the PyKale 0.2.2 maintainers' edits: sorted, cleaned, no release pull request."""
    body = update_changelog.drop_empty_groups(update_changelog.format_entries(RELEASE_022_GENERATED))

    assert body == RELEASE_022_FORMATTED


def test_format_entries_sorts_numerically_and_drops_duplicates():
    """Tests that entries sort by number rather than text, and a pull request listed twice appears once."""
    generated = (
        "* [#100](https://example.org/100): Third\n"
        "* [#9](https://example.org/9): First\n"
        "* [#100](https://example.org/100): Duplicate\n"
        "* [#20](https://example.org/20): Second\n"
    )

    assert update_changelog.format_entries(generated) == (
        "* [#9](https://example.org/9): First\n"
        "* [#20](https://example.org/20): Second\n"
        "* [#100](https://example.org/100): Third"
    )


def test_updater_writes_formatted_section(tmp_path, monkeypatch):
    """Tests that the command-line interface writes the formatted section and comment."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(HISTORY.encode())
    generated = tmp_path / "generated.md"
    generated.write_text(RELEASE_022_GENERATED, encoding="utf-8")

    assert run_updater(monkeypatch, "0.2.2", generated, changelog) == 0

    assert changelog.read_bytes().decode() == f"# Version  0.2.2\n\n{RELEASE_022_FORMATTED}\n\n{HISTORY}"
    assert generated.read_text(encoding="utf-8") == f"# Version  0.2.2\n\n{RELEASE_022_FORMATTED}\n"


# release_notes.py


def test_extract_section_stops_at_next_version():
    """Tests that a version's notes are its own section only, found with any header spacing or leading v."""
    changelog = f"# Version 0.3.0\n\n{RELEASE_022_FORMATTED}\n\n{HISTORY}"

    assert release_notes.extract_section(changelog, "v0.3.0") == RELEASE_022_FORMATTED
    assert release_notes.extract_section(changelog, "0.1.2") == (
        "#### Bug Fixes\n\n* [#389](https://github.com/pykale/pykale/pull/389): Fix something"
    )
    assert release_notes.extract_section(changelog, "0.4.0") is None


@pytest.mark.parametrize("content", [None, "", HISTORY, "# Version  0.3.0\n\n"])
def test_release_notes_missing_section_fails(tmp_path, monkeypatch, capsys, content):
    """Tests that a missing changelog, or a missing or empty section, fails so nothing is released without notes."""
    changelog = tmp_path / "CHANGELOG.md"
    if content is not None:
        changelog.write_text(content, encoding="utf-8")
    output = tmp_path / "notes.md"
    monkeypatch.setattr("sys.argv", ["release_notes.py", "0.3.0", str(changelog), str(output)])

    assert release_notes.main() == 1
    assert "::error::" in capsys.readouterr().out
    assert not output.exists()


def test_release_notes_main_writes_section(tmp_path, monkeypatch):
    """Tests that the command-line interface writes the section body for the release notes."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(f"# Version  0.3.0rc1\n\n{RELEASE_022_FORMATTED}\n\n{HISTORY}", encoding="utf-8")
    output = tmp_path / "notes.md"
    monkeypatch.setattr("sys.argv", ["release_notes.py", "0.3.0RC1", str(changelog), str(output)])

    assert release_notes.main() == 0
    assert output.read_text(encoding="utf-8") == RELEASE_022_FORMATTED + "\n"


# release and project workflows

WORKFLOWS = ROOT / ".github" / "workflows"


def test_release_runs_only_for_approved_release_pull_requests_from_maintainers():
    """Guards who can trigger a merge and a PyPI release."""
    workflow = (WORKFLOWS / "release-on-approval.yml").read_text(encoding="utf-8")

    assert "pull_request_review:" in workflow
    assert "github.event.review.state == 'approved'" in workflow
    assert 'contains(fromJSON(\'["OWNER", "MEMBER", "COLLABORATOR"]\'), github.event.review.author_association)' in (
        workflow
    )
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "github.event.pull_request.base.ref == github.event.repository.default_branch" in workflow
    assert '"$REVIEWED_SHA" != "$HEAD_SHA"' in workflow


def test_release_uses_trusted_scripts_and_checks_before_merging():
    """Guards the order: version and notes from base-commit scripts, passing CI, then a merge of the approved commit."""
    workflow = (WORKFLOWS / "release-on-approval.yml").read_text(encoding="utf-8")

    assert workflow.count("persist-credentials: false") == 2
    assert "ref: ${{ github.event.pull_request.base.sha }}" in workflow
    assert 'python3 "$RUNNER_TEMP/release_version.py" kalecancer/__init__.py' in workflow
    assert 'python3 "$RUNNER_TEMP/release_notes.py"' in workflow
    assert 'cp trusted-base/.github/scripts/release_checks.py "$RUNNER_TEMP/release_checks.py"' in workflow
    assert 'python3 "$RUNNER_TEMP/release_checks.py" "$WORKFLOW_NAME"' in workflow
    assert '[ "$status" -ne 3 ]' in workflow
    assert 'gh api -X PUT "repos/$REPOSITORY/pulls/$PR_NUMBER/merge"' in workflow
    assert '-f sha="$HEAD_SHA" -f merge_method=merge' in workflow
    assert "select(.merged == true) | .sha" in workflow
    order = [
        'release_notes.py" "$VERSION"',
        'gh release view "v$VERSION"',
        "Wait for the other checks",
        "gh api -X PUT",
        "gh release create",
        "uses: ./",
    ]
    positions = [workflow.index(step) for step in order]
    assert positions == sorted(positions)


@pytest.mark.parametrize("name", ["release.yml", "release-on-approval.yml"])
def test_release_workflows_use_trusted_publishing(name):
    """Tests that both release paths build through build-dist.yml and publish from a top-level job without tokens.

    Trusted publishing cannot run inside a reusable workflow, so the publish jobs must be in the workflow files that
    pypi.org and test.pypi.org trust, each with id-token: write and the environment configured there.
    """
    workflow = (WORKFLOWS / name).read_text(encoding="utf-8")

    assert "uses: ./.github/workflows/build-dist.yml" in workflow
    assert workflow.count("id-token: write") == 2
    assert workflow.count("uses: pypa/gh-action-pypi-publish@release/v1") == 2
    assert "name: testpypi" in workflow and "repository-url: https://test.pypi.org/legacy/" in workflow
    assert "name: pypi" in workflow
    assert "password:" not in workflow and "PYPI_API_TOKEN" not in workflow
    assert "secrets: inherit" not in workflow


def test_manual_release_runs_when_a_release_is_published():
    """Publishing a draft release must trigger the manual release path."""
    workflow = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "types: [published]" in workflow


def test_build_workflow_only_builds():
    """Tests that the reusable workflow uploads the distribution and never publishes, which it could not do there."""
    workflow = (WORKFLOWS / "build-dist.yml").read_text(encoding="utf-8")

    assert "workflow_call:" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "uses: pypa/gh-action-pypi-publish" not in workflow
    assert "id-token: write" not in workflow


def test_approved_release_publishes_the_merge_commit():
    """Tests that the approval workflow builds the merge commit and picks the index from the version."""
    workflow = (WORKFLOWS / "release-on-approval.yml").read_text(encoding="utf-8")

    assert "ref: ${{ needs.merge-and-tag.outputs.sha }}" in workflow
    assert "if: needs.merge-and-tag.outputs.prerelease == 'true'" in workflow
    assert "if: needs.merge-and-tag.outputs.prerelease == 'false'" in workflow


# Releasing a pull request a maintainer merged by hand before the approval run started

APPROVED = "a" * 40
LATER = "b" * 40
MERGE_COMMIT = "c" * 40


def workflow_step(name):
    """Returns a step of the release-on-approval merge-and-tag job, found by its name."""
    workflow = yaml.safe_load((WORKFLOWS / "release-on-approval.yml").read_text(encoding="utf-8"))
    return next(step for step in workflow["jobs"]["merge-and-tag"]["steps"] if step["name"] == name)


def run_step(tmp_path, name, gh, **env):
    """Runs a workflow step's script with bash -e, as GitHub does, and with ``gh`` replaced by a shell function.

    Args:
        tmp_path (Path): The working directory, which also holds the step's output file.
        name (str): The step's name.
        gh (str): The body of the fake ``gh`` function.
        **env (str): The step's environment.

    Returns:
        tuple[int, dict[str, str], str]: The exit status, the outputs the step wrote and its standard output.
    """
    script = f"gh() {{\n{gh}\n}}\n{workflow_step(name)['run']}"
    environment = {**os.environ, "GITHUB_OUTPUT": "outputs", **env}
    result = subprocess.run(
        [shutil.which("bash"), "-e", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output_file = tmp_path / "outputs"
    lines = output_file.read_text(encoding="utf-8").splitlines() if output_file.exists() else []
    return result.returncode, dict(line.split("=", 1) for line in lines), result.stdout


STATE_STEP = "Require approval of the pull request's latest commit"
STATE_ENV = {"PR_NUMBER": "22", "REPOSITORY": "pykale/cancer", "HEAD_SHA": APPROVED}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize(
    "state, head, merge_commit, reviewed, exit_code, outputs",
    [
        ("OPEN", APPROVED, "", APPROVED, 0, {"approved": "true"}),
        # Merged by hand at the approved commit: released from the merge commit.
        ("MERGED", APPROVED, MERGE_COMMIT, APPROVED, 0, {"merged_sha": MERGE_COMMIT, "approved": "true"}),
        # Merged by hand after another push: that commit was never approved, so nothing is released.
        ("MERGED", LATER, MERGE_COMMIT, APPROVED, 0, {}),
        ("CLOSED", APPROVED, "", APPROVED, 0, {}),
        # The head moved after the approval, or an older commit was approved.
        ("OPEN", LATER, "", APPROVED, 1, {}),
        ("OPEN", APPROVED, "", LATER, 1, {}),
    ],
)
def test_release_state_accepts_only_the_approved_head(
    tmp_path, state, head, merge_commit, reviewed, exit_code, outputs
):
    """Tests which pull request states start a release, including one merged by hand before the run started."""
    gh = f"printf '%s\t%s\t%s\n' {state} {head} '{merge_commit}'"

    status, written, _ = run_step(tmp_path, STATE_STEP, gh, REVIEWED_SHA=reviewed, **STATE_ENV)

    assert (status, written) == (exit_code, outputs)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_release_state_fails_when_gh_fails(tmp_path):
    """Tests that a failed pull request lookup fails the run instead of being read as an unapproved head."""
    status, written, _ = run_step(tmp_path, STATE_STEP, "return 1", REVIEWED_SHA=APPROVED, **STATE_ENV)

    assert status != 0
    assert written == {}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize("released, outputs", [(True, {}), (False, {"version": "0.2.0"})])
def test_release_stops_when_the_version_is_already_released(tmp_path, released, outputs):
    """Tests that a second approval of a released pull request stops before merging or releasing again."""
    gh = f'[ "$*" = "release view v0.2.0 --repo pykale/cancer" ] && return {0 if released else 1}; return 99'

    status, written, _ = run_step(
        tmp_path, "Stop if this version is already released", gh, VERSION="0.2.0", REPOSITORY="pykale/cancer"
    )

    assert (status, written) == (0, outputs)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_hand_merged_release_uses_the_merge_commit_without_merging(tmp_path):
    """Tests that a pull request merged by hand is released from its merge commit, with no second merge."""
    status, written, _ = run_step(
        tmp_path, "Merge the release pull request", "return 99", MERGED_SHA=MERGE_COMMIT, **STATE_ENV
    )

    assert (status, written) == (0, {"sha": MERGE_COMMIT})


def test_changelog_commit_runs_ci():
    """Tests that CI runs on the changelog commit, which the approval workflow requires before merging."""
    changelog = WORKFLOW.read_text(encoding="utf-8")
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "actions: write" in changelog
    assert 'gh workflow run ci.yml --repo "$REPOSITORY" --ref "$HEAD_REF"' in changelog
    assert "if: steps.commit.outputs.pushed == 'true'" in changelog
    assert "workflow_dispatch:" in ci


def test_project_board_is_kalecancer_and_never_given_to_forks():
    """Tests that issues and same-repository pull requests go to the KaleCancer board, not PyKale's."""
    workflow = (WORKFLOWS / "project.yml").read_text(encoding="utf-8")

    assert workflow.count("project-url: https://github.com/orgs/pykale/projects/14") == 2
    assert "projects/4" not in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow


def run(workflow, status="completed", conclusion="success"):
    """Returns a workflow run as ``gh run list --json workflowName,status,conclusion`` gives it."""
    return {"workflowName": workflow, "status": status, "conclusion": conclusion if status == "completed" else ""}


OWN = "release-on-approval"


@pytest.mark.parametrize(
    "runs, expected",
    [
        ([run("CI"), run("codeql-analysis")], "passed"),
        ([run("CI", conclusion="skipped")], "passed"),
        ([run("CI", conclusion="failure"), run("CI")], "failed"),
        ([run("CI", conclusion="cancelled")], "failed"),
        ([run("CI", conclusion="action_required"), run("CI", conclusion="failure")], "failed"),
        ([run("CI", conclusion="action_required"), run("CI")], "passed"),
        ([run("CI", conclusion="action_required"), run("CI", status="in_progress")], "running"),
        ([run("CI", conclusion="action_required")], "awaiting approval"),
    ],
)
def test_release_checks_judge_each_workflow_as_a_whole(runs, expected):
    """A run waiting for approval never ran, so it neither fails its workflow nor counts as a pass."""
    assert load("release_checks").workflow_state(runs) == expected


def test_release_checks_pass_bot_commit_once_dispatched_ci_passed():
    """The changelog commit's own pull_request runs wait for approval, but the CI run changelog.yml dispatches passed.

    This is what an approval of an unedited changelog commit sees, e.g. 4f45e8d on pykale/cancer#54. CodeQL has no
    other run, so the release waits for it to be approved and run instead of failing.
    """
    checks = load("release_checks")
    runs = [
        run("CI", conclusion="action_required"),
        run("codeql-analysis", conclusion="action_required"),
        run("CI"),
        run(OWN, status="in_progress"),
    ]

    status, message = checks.summarise(runs, OWN)

    assert status == checks.EXIT_PENDING
    assert "waiting for approval: codeql-analysis" in message
    assert "CI" not in message.split("waiting for approval:")[1].split("(")[0]

    status, message = checks.summarise([*runs[:1], run("codeql-analysis"), *runs[2:]], OWN)
    assert (status, message) == (checks.EXIT_PASSED, "All checks passed.")


def test_release_checks_report_failures_before_pending_runs():
    checks = load("release_checks")
    runs = [run("CI", status="queued"), run("codeql-analysis", conclusion="failure")]

    status, message = checks.summarise(runs, OWN)

    assert status == checks.EXIT_FAILED
    assert message == "Checks failed: codeql-analysis. Fix them, push, and approve again."


@pytest.mark.parametrize(
    "runs, waiting_for",
    [
        ([], "no CI run has succeeded yet"),
        ([run("codeql-analysis")], "no CI run has succeeded yet"),
        ([run("CI", status="in_progress")], "running: CI"),
        ([run("CI", conclusion="skipped")], "no CI run has succeeded yet"),
        ([run("CI", conclusion="neutral")], "no CI run has succeeded yet"),
        ([run("CI", conclusion="skipped"), run("CI", conclusion="neutral")], "no CI run has succeeded yet"),
    ],
)
def test_release_checks_require_ci(runs, waiting_for):
    """The release never goes ahead without a CI run that succeeded, even when nothing else is pending.

    A skipped or neutral CI run never ran the tests, so, as before this script, only "success" counts for CI.
    """
    checks = load("release_checks")
    status, message = checks.summarise(runs, OWN)
    assert status == checks.EXIT_PENDING
    assert waiting_for in message


def test_release_checks_ignore_their_own_run():
    checks = load("release_checks")
    assert checks.summarise([run("CI"), run(OWN, status="in_progress")], OWN)[0] == checks.EXIT_PASSED


@pytest.mark.parametrize(
    "runs, exit_code, output",
    [
        ([run("CI")], 0, "All checks passed.\n"),
        ([run("CI", conclusion="failure")], 1, "Checks failed: CI. Fix them, push, and approve again.\n"),
        ([run("CI", conclusion="action_required")], 3, "Waiting: waiting for approval: CI"),
    ],
)
def test_release_checks_main(monkeypatch, capsys, runs, exit_code, output):
    """Tests the exit status the workflow's wait loop acts on: 0 to merge, 3 to keep waiting, anything else to stop."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(runs)))
    assert load("release_checks").main([OWN]) == exit_code
    assert capsys.readouterr().out.startswith(output)


def test_release_checks_usage_error(capsys):
    assert load("release_checks").main([]) == 2
    assert "usage:" in capsys.readouterr().err
