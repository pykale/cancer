"""Architectural boundary checks for ``kalecancer.survival``."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SURVIVAL_ROOT = Path(__file__).resolve().parents[1] / "kalecancer" / "survival"
ALLOWED_KALECANCER_PREFIX = "kalecancer.survival"


@dataclass(frozen=True)
class ImportViolation:
    path: Path
    line: int
    imported_name: str

    def message(self) -> str:
        rel_path = self.path.relative_to(SURVIVAL_ROOT.parents[1])
        return f"{rel_path}:{self.line}: {self.imported_name}"


def _is_allowed_kalecancer_module(module: str) -> bool:
    return module == ALLOWED_KALECANCER_PREFIX or module.startswith(f"{ALLOWED_KALECANCER_PREFIX}.")


def _check_import(node: ast.Import, path: Path) -> list[ImportViolation]:
    violations: list[ImportViolation] = []
    for alias in node.names:
        name = alias.name
        if name == "kalecancer" or (
            name.startswith("kalecancer.") and not _is_allowed_kalecancer_module(name)
        ):
            violations.append(ImportViolation(path, node.lineno, name))
    return violations


def _check_import_from(node: ast.ImportFrom, path: Path) -> list[ImportViolation]:
    violations: list[ImportViolation] = []

    if node.level >= 2:
        module = node.module or ""
        suffix = f"{module}" if module else ""
        imported_name = "." * node.level + suffix
        violations.append(ImportViolation(path, node.lineno, imported_name))
        return violations

    if node.level == 0 and node.module is not None:
        module = node.module
        if module == "kalecancer" or (
            module.startswith("kalecancer.") and not _is_allowed_kalecancer_module(module)
        ):
            violations.append(ImportViolation(path, node.lineno, module))

    return violations


def _collect_violations(path: Path) -> list[ImportViolation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[ImportViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(_check_import(node, path))
        elif isinstance(node, ast.ImportFrom):
            violations.extend(_check_import_from(node, path))

    return violations


def test_survival_does_not_import_other_kalecancer_modules() -> None:
    """``kalecancer.survival`` must remain isolated from the rest of the package."""
    py_files = sorted(SURVIVAL_ROOT.rglob("*.py"))
    assert py_files, f"expected Python files under {SURVIVAL_ROOT}"

    violations: list[ImportViolation] = []
    for path in py_files:
        violations.extend(_collect_violations(path))

    if violations:
        details = "\n".join(violation.message() for violation in violations)
        raise AssertionError(
            "kalecancer/survival must not import from elsewhere in kalecancer "
            "(relative imports with level >= 2 are also forbidden):\n"
            f"{details}"
        )
