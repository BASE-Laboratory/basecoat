"""Declarative repository layer linter backed by ``[tool.basecoat.layers]`` in ``pyproject.toml``.

Design rule: ``basecoat`` never hardcodes downstream package names. Each repository
declares its own layer number, source directories, and banned top-level imports in
its ``pyproject.toml``::

    [tool.basecoat.layers]
    layer = 2
    source_dirs = ["src"]
    banned_imports = ["multimodal_benchmark", "openimpala", "x_fuse"]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import BasecoatError


@dataclass(frozen=True)
class LayerConfig:
    """Parsed ``[tool.basecoat.layers]`` configuration for a repository."""

    layer: int
    banned_imports: tuple[str, ...]
    source_dirs: tuple[str, ...] = ("src",)


@dataclass(frozen=True)
class LayerViolation:
    """Single banned import detected in a repository source file."""

    file: Path
    line: int
    imported_module: str
    banned_root: str

    def format(self, repo_root: Path | None = None) -> str:
        rel = self.file.relative_to(repo_root) if repo_root is not None else self.file
        return (
            f"{rel}:{self.line}: banned import '{self.imported_module}' "
            f"(matches banned root '{self.banned_root}')"
        )


def _parse_pyproject_layers_fallback(text: str) -> dict[str, Any]:
    """Dependency-free parser for ``[tool.basecoat.layers]`` on Python 3.10 without ``tomllib``."""
    in_section = False
    section_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip()
            if header == "tool.basecoat.layers":
                in_section = True
                continue
            if in_section:
                break
        elif in_section:
            section_lines.append(stripped)

    if not section_lines:
        return {}

    joined = "\n".join(section_lines)
    out: dict[str, Any] = {}
    m_layer = re.search(r"(?m)^layer\s*=\s*(\d+)\s*$", joined)
    if m_layer:
        out["layer"] = int(m_layer.group(1))

    for key in ("banned_imports", "source_dirs"):
        m_arr = re.search(rf"{key}\s*=\s*\[(.*?)\]", joined, flags=re.DOTALL)
        if m_arr:
            items = re.findall(r"""['"]([^'"]+)['"]""", m_arr.group(1))
            out[key] = items
    return out


def load_layer_config(pyproject_path: str | Path) -> LayerConfig:
    """Load ``[tool.basecoat.layers]`` from *pyproject_path*."""
    p = Path(pyproject_path)
    if not p.is_file():
        raise FileNotFoundError(f"pyproject.toml not found at '{p}'.")
    text = p.read_text(encoding="utf-8")

    section: dict[str, Any]
    if sys.version_info >= (3, 11):
        import tomllib

        doc = tomllib.loads(text)
        section = doc.get("tool", {}).get("basecoat", {}).get("layers", {})
    else:  # pragma: no cover - exercised on Python 3.10 CI runners
        section = _parse_pyproject_layers_fallback(text)

    if not isinstance(section, dict) or "layer" not in section:
        raise ValueError(f"Missing [tool.basecoat.layers] with integer 'layer' in '{p}'.")

    layer_num = int(section["layer"])
    if layer_num not in (1, 2, 3, 4):
        raise ValueError(f"[tool.basecoat.layers].layer must be 1, 2, 3, or 4, got {layer_num}.")

    raw_banned = section.get("banned_imports", [])
    raw_dirs = section.get("source_dirs", ["src"])
    if not isinstance(raw_banned, list) or not isinstance(raw_dirs, list):
        raise ValueError("banned_imports and source_dirs must be lists of strings.")

    return LayerConfig(
        layer=layer_num,
        banned_imports=tuple(str(x).strip() for x in raw_banned if str(x).strip()),
        source_dirs=tuple(str(d).strip() for d in raw_dirs if str(d).strip()),
    )


def check_layer_imports(
    repo_root: str | Path,
    *,
    pyproject_path: str | Path | None = None,
    raise_on_violation: bool = False,
) -> list[LayerViolation]:
    """Scan Python modules under configured ``source_dirs`` for banned ecosystem imports."""
    root = Path(repo_root).resolve()
    cfg_path = (
        Path(pyproject_path).resolve() if pyproject_path is not None else (root / "pyproject.toml")
    )
    cfg = load_layer_config(cfg_path)
    banned_set = set(cfg.banned_imports)
    if not banned_set:
        return []

    violations: list[LayerViolation] = []
    for rel_dir in cfg.source_dirs:
        target = root / rel_dir
        if not target.exists():
            continue
        py_files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".", 1)[0]
                        if top in banned_set:
                            violations.append(
                                LayerViolation(
                                    file=py_file,
                                    line=node.lineno,
                                    imported_module=alias.name,
                                    banned_root=top,
                                )
                            )
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    top = node.module.split(".", 1)[0]
                    if top in banned_set:
                        violations.append(
                            LayerViolation(
                                file=py_file,
                                line=node.lineno,
                                imported_module=node.module,
                                banned_root=top,
                            )
                        )

    if raise_on_violation and violations:
        formatted = "\n  - ".join(v.format(root) for v in violations)
        raise BasecoatError(
            f"Layer {cfg.layer} import boundary violated ({len(violations)} offending import(s)):\n  - {formatted}"
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (`python -m basecoat.layers [repo_root]`)."""
    parser = argparse.ArgumentParser(
        prog="python -m basecoat.layers",
        description="Verify [tool.basecoat.layers] import boundaries in a repository.",
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Repository root containing pyproject.toml (default: current directory).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.repo_root).resolve()
    cfg = load_layer_config(root / "pyproject.toml")
    violations = check_layer_imports(root)
    if violations:
        for v in violations:
            print(v.format(root), file=sys.stderr)
        return 1
    print(
        f"[basecoat.layers] OK: layer={cfg.layer}, checked {list(cfg.source_dirs)}, "
        f"0 banned imports ({len(cfg.banned_imports)} rule(s))."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
