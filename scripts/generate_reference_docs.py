#!/usr/bin/env python3
"""Generates ``docs/REFERENCE.md``: a per-file, per-function/class API
reference, extracted directly from source docstrings and signatures via
the ``ast`` module (so it can't drift out of sync with argument names).

Run whenever the code changes and you want the reference doc refreshed::

    python scripts/generate_reference_docs.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRS = ["stock_predictor", "scripts", "tests"]
OUTPUT_PATH = ROOT / "docs" / "REFERENCE.md"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(node.args.args, defaults):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        args.append(piece)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    for kwarg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        piece = kwarg.arg
        if kwarg.annotation is not None:
            piece += f": {ast.unparse(kwarg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        args.append(piece)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({', '.join(args)}){returns}"


def _doc_summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.strip()


def _document_function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str = "") -> list[str]:
    lines = [f"{indent}#### `{node.name}{_signature(node)}`"]
    doc = _doc_summary(node)
    if doc:
        lines.append("")
        lines.append(f"{indent}{doc}")
    lines.append("")
    return lines


def _document_class(node: ast.ClassDef) -> list[str]:
    lines = [f"### `class {node.name}`"]
    doc = _doc_summary(node)
    if doc:
        lines.append("")
        lines.append(doc)
    lines.append("")
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines += _document_function(item)
    return lines


def _document_module(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(ROOT).as_posix()

    lines = [f"## `{rel}`", ""]
    module_doc = _doc_summary(tree)
    if module_doc:
        lines.append(module_doc)
        lines.append("")

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines += _document_class(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines += _document_function(node)

    return lines


def main() -> None:
    all_lines = [
        "# API Reference",
        "",
        "Auto-generated from source docstrings by "
        "`scripts/generate_reference_docs.py` -- do not hand-edit; re-run "
        "the script instead. See `docs/ARCHITECTURE.md` for the narrative "
        "version with diagrams.",
        "",
    ]

    for package_dir in PACKAGE_DIRS:
        for path in sorted((ROOT / package_dir).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            all_lines += _document_module(path)

    OUTPUT_PATH.write_text("\n".join(all_lines))
    print(f"Wrote {OUTPUT_PATH} ({len(all_lines)} lines)")


if __name__ == "__main__":
    main()
