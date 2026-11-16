#!/usr/bin/env python3
"""Decide whether the checks on an approved release commit allow the release to go ahead.

Usage: gh run list --commit <sha> --json workflowName,status,conclusion | release_checks.py <own-workflow-name>

Runs are judged per workflow. A workflow fails when any of its runs failed. Otherwise it is still running while any
run has not finished, and it passes when a run succeeded, was skipped or was neutral. A run that ended
"action_required" never ran: GitHub is waiting for someone to approve it, which happens to the runs a bot's commit,
such as the changelog commit, starts. Such a run doesn't count against its workflow when another run of the same
workflow passed, e.g. the CI run changelog.yml dispatches. A workflow whose only runs wait for approval is reported
as waiting, not failed, so the release waits until a maintainer approves and the runs pass.

The release also requires a passing "CI" run.

Exit status: 0 when every check passed, 1 when a check failed, 3 while checks are running or waiting for approval.
A one-line summary is printed in each case.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable

PASSED = {"success", "skipped", "neutral"}
AWAITING_APPROVAL = "action_required"
REQUIRED_WORKFLOW = "CI"

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_PENDING = 3


def workflow_state(runs: Iterable[dict]) -> str:
    """Returns the state of one workflow from its runs on a commit.

    Args:
        runs (Iterable[dict]): The workflow's runs, each with "status" and "conclusion" as ``gh run list`` gives them.

    Returns:
        str: "failed", "running", "passed" or "awaiting approval".
    """
    runs = list(runs)
    finished = [run for run in runs if run.get("status") == "completed"]
    if any(run.get("conclusion") not in PASSED | {AWAITING_APPROVAL} for run in finished):
        return "failed"
    if len(finished) < len(runs):
        return "running"
    if any(run.get("conclusion") in PASSED for run in finished):
        return "passed"
    return "awaiting approval"


def summarise(runs: Iterable[dict], own_workflow: str) -> tuple[int, str]:
    """Decides whether the release can go ahead, from every workflow run on the approved commit.

    Args:
        runs (Iterable[dict]): The runs, each with "workflowName", "status" and "conclusion".
        own_workflow (str): The name of the release workflow, whose own run is left out.

    Returns:
        tuple[int, str]: The exit status (``EXIT_PASSED``, ``EXIT_FAILED`` or ``EXIT_PENDING``) and a summary.
    """
    by_workflow: dict[str, list[dict]] = {}
    for run in runs:
        name = run.get("workflowName", "")
        if name != own_workflow:
            by_workflow.setdefault(name, []).append(run)

    states = {name: workflow_state(workflow_runs) for name, workflow_runs in sorted(by_workflow.items())}
    failed = [name for name, state in states.items() if state == "failed"]
    running = [name for name, state in states.items() if state == "running"]
    waiting = [name for name, state in states.items() if state == "awaiting approval"]

    if failed:
        return EXIT_FAILED, f"Checks failed: {', '.join(failed)}. Fix them, push, and approve again."

    pending = []
    if running:
        pending.append(f"running: {', '.join(running)}")
    if waiting:
        pending.append(
            f"waiting for approval: {', '.join(waiting)} (approve and run them from the pull request's checks)"
        )
    if states.get(REQUIRED_WORKFLOW) != "passed" and REQUIRED_WORKFLOW not in running + waiting:
        pending.append(f"no {REQUIRED_WORKFLOW} run has passed yet")
    if pending:
        return EXIT_PENDING, "Waiting: " + "; ".join(pending) + "."

    return EXIT_PASSED, "All checks passed."


def main(argv: list[str] | None = None) -> int:
    """Reads the runs as JSON from standard input, prints the summary and returns the exit status."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.stderr.write("usage: release_checks.py <own-workflow-name> < runs.json\n")
        return 2
    status, message = summarise(json.load(sys.stdin), argv[0])
    sys.stdout.write(f"{message}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
