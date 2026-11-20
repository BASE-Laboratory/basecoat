"""AST enforcement that zarr/numcodecs live exclusively inside basecoat.io.ome_zarr."""

from __future__ import annotations

import ast
from pathlib import Path

BANNED_RUNTIME_IMPORTS = {
    "torch",
    "skimage",
    "dask",
    "ome_zarr",
    "ome_zarr_models",
    "PIL",
    "mpi4py",
}


def _imported_top_packages(py_path: Path) -> list[tuple[int, str, bool]]:
    """Return (lineno, top_level_pkg, is_module_top_level) for all imports in *py_path*."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    top_nodes = set(tree.body)
    found: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        is_top = node in top_nodes
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name.split(".")[0], is_top))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module.split(".")[0], is_top))
    return found


def test_zarr_isolated_to_single_module_and_no_heavy_deps() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "basecoat"
    ome_zarr_mod = (src_root / "io" / "ome_zarr.py").resolve()
    assert ome_zarr_mod.is_file()

    py_files = sorted(src_root.rglob("*.py"))
    assert len(py_files) >= 8

    for py_file in py_files:
        resolved = py_file.resolve()
        for lineno, pkg, is_top in _imported_top_packages(resolved):
            if pkg in ("zarr", "numcodecs"):
                assert resolved == ome_zarr_mod, (
                    f"{resolved}:{lineno} imports '{pkg}', violating single-module zarr isolation "
                    f"(allowed only in io/ome_zarr.py)."
                )
            assert pkg not in BANNED_RUNTIME_IMPORTS, (
                f"{resolved}:{lineno} imports banned heavy dependency '{pkg}'."
            )
            if pkg == "SimpleITK":
                assert not is_top, (
                    f"{resolved}:{lineno} imports SimpleITK at module top-level; must stay lazy behind basecoat[sitk]."
                )
