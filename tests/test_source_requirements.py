"""Structural subject checks supplement behavioral tests and peer review."""

import ast
from pathlib import Path
import re


SOURCE = Path(__file__).resolve().parents[1] / "src" / "fly_in"


def test_source_functions_and_classes_have_docstrings() -> None:
    """Include private methods and nested helpers in the documentation gate."""
    missing = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                if ast.get_docstring(node):
                    continue
                missing.append(
                    f"{path.relative_to(SOURCE)}:{node.lineno} {node.name}"
                )
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_graph_and_planning_behavior_belongs_to_objects() -> None:
    """Flag module-level domain behavior throughout the application.

    Thin main entry adapters and test helpers are outside this check.
    Class placement alone cannot establish sound object-oriented design;
    ownership, composition, and responsibilities still require peer review.
    """
    standalone = []
    for package in (SOURCE, SOURCE.parents[1] / "scripts"):
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name != "main"):
                    standalone.append(
                        f"{path.relative_to(SOURCE.parents[1])}:"
                        f"{node.lineno} {node.name}"
                    )
    assert not standalone, (
        "Graph/planning functions need object owners:\n"
        + "\n".join(standalone)
    )


def test_source_does_not_import_graph_libraries() -> None:
    """Catch direct imports of graph packages forbidden by the subject."""
    forbidden = {"networkx", "graphlib", "igraph", "rustworkx", "graph_tool"}
    violations = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.split(".")[0] in forbidden:
                    violations.append(
                        f"{path.relative_to(SOURCE)}:{node.lineno} {module}"
                    )
    assert not violations, "Forbidden imports:\n" + "\n".join(violations)


def test_local_markdown_links_point_to_existing_files() -> None:
    """Keep README and compliance evidence navigable after docs move."""
    root = SOURCE.parents[1]
    documents = [
        root / "README.md",
        *sorted((root / "docs").glob("*.md")),
        *sorted((root / "maps").glob("README.md")),
    ]
    broken = []
    for document in documents:
        contents = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", contents):
            destination = target.split("#", 1)[0]
            if (not destination or "://" in destination
                    or destination.startswith(("mailto:", "tel:"))):
                continue
            path = (document.parent / destination).resolve()
            if not path.is_file():
                broken.append(
                    f"{document.relative_to(root)}: {target}"
                )
    details = "\n".join(broken)
    assert not broken, "Broken local documentation links:\n" + details
