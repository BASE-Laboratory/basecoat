"""Tests for Savu out-of-core HDF5 reader, SimilarityTransform3D chaining, and layer linter (v0.2.0)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from basecoat import (
    BasecoatError,
    SimilarityTransform3D,
    SpaceMismatchError,
    check_layer_imports,
    open_savu_h5,
    read_savu_subvolume,
)
from basecoat.layers import _parse_pyproject_layers_fallback
from basecoat.layers import main as layers_cli_main


def test_savu_xzy_lazy_and_subvolume_reader(tmp_path: Path) -> None:
    savu_path = tmp_path / "tomo_p4_tomopy_recon.h5"
    # Savu TomopyRecon disk order: (X=24, Z_vertical=14, Y=24) with shape[0] == shape[2] != shape[1]
    raw_xzy = np.arange(24 * 14 * 24, dtype=np.float32).reshape(24, 14, 24)
    with h5py.File(savu_path, "w") as fh:
        fh.create_dataset("4-TomopyRecon-tomo/data", data=raw_xzy)

    with open_savu_h5(savu_path, voxel_size_um=1.625, space="i13_169067") as lv:
        assert lv.shape == (14, 24, 24)
        assert lv.axis_names == ("z", "y", "x")
        assert lv.provenance["savu_xzy_transposed"] is True
        sub = lv.read_region((slice(2, 7), slice(4, 10), slice(5, 13)))
        expected = np.transpose(raw_xzy[5:13, 2:7, 4:10], (1, 2, 0))
        np.testing.assert_allclose(sub.data, expected)
        assert sub.origin_in("um") == pytest.approx((2 * 1.625, 4 * 1.625, 5 * 1.625))

    vol_flipped = read_savu_subvolume(
        savu_path,
        zyx_slices=(slice(2, 6), slice(3, 9), slice(4, 12)),
        stride=2,
        flip_x=True,
        voxel_size_um=0.54,
        dtype=np.float64,
    )
    expected_strided = np.flip(np.transpose(raw_xzy[4:12:2, 2:6:2, 3:9:2], (1, 2, 0)), axis=2)
    assert vol_flipped.dtype == np.float64
    np.testing.assert_allclose(vol_flipped.data, expected_strided)
    assert vol_flipped.spacing_in("um") == pytest.approx((1.08, 1.08, 1.08))
    # Former last x-cell of slice(4, 12, 2) -> indices [4, 6, 8, 10] -> index 10 at 10 * 0.54 = 5.4 um
    assert vol_flipped.origin_in("um") == pytest.approx((2 * 0.54, 3 * 0.54, 10 * 0.54))


def test_cropped_non_savu_hdf5_reads_without_transposition(tmp_path: Path) -> None:
    crop_path = tmp_path / "169067_8bit_cropped.h5"
    raw_zyx = np.arange(10 * 16 * 16, dtype=np.uint8).reshape(10, 16, 16)
    with h5py.File(crop_path, "w") as fh:
        fh.create_dataset("data", data=raw_zyx)

    sub = read_savu_subvolume(
        crop_path,
        zyx_slices=(slice(1, 5), slice(2, 8), slice(3, 9)),
        voxel_size_um=1.625,
        dtype=np.float32,
    )
    assert sub.provenance["savu_xzy_transposed"] is False
    np.testing.assert_allclose(sub.data, raw_zyx[1:5, 2:8, 3:9].astype(np.float32))


def test_similarity_transform_then_similarity_three_rung_chain() -> None:
    t1 = SimilarityTransform3D.from_euler_xyz(
        euler_xyz_rad=(0.01, -0.02, 0.015),
        scale=0.996,
        translation_xyz=(1.2, -0.8, 2.1),
        center_xyz=(100.0, 120.0, 80.0),
        from_space="r0",
        to_space="r1",
    )
    t2 = SimilarityTransform3D.from_euler_xyz(
        euler_xyz_rad=(-0.005, 0.012, -0.008),
        scale=0.987,
        translation_xyz=(-3.4, 1.9, 0.7),
        center_xyz=(90.0, 110.0, 75.0),
        from_space="r1",
        to_space="r2",
    )
    t3 = SimilarityTransform3D.from_euler_xyz(
        euler_xyz_rad=(0.008, 0.004, -0.011),
        scale=0.998,
        translation_xyz=(0.5, -1.1, -1.9),
        center_xyz=(95.0, 105.0, 70.0),
        from_space="r2",
        to_space="r3",
    )

    chained = t1.then_similarity(t2).then_similarity(t3)
    assert chained.from_space == "r0"
    assert chained.to_space == "r3"
    assert chained.scale == pytest.approx(t1.scale * t2.scale * t3.scale)

    pts = np.array([[10.0, 25.0, 40.0], [130.0, 95.0, 160.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    seq = t3.transform_points(t2.transform_points(t1.transform_points(pts)))
    np.testing.assert_allclose(chained.transform_points(pts), seq, atol=1e-11)
    np.testing.assert_allclose(
        chained.to_affine().matrix_4x4,
        t1.to_affine().then(t2.to_affine()).then(t3.to_affine()).matrix_4x4,
        atol=1e-11,
    )

    with pytest.raises(SpaceMismatchError):
        t1.then_similarity(t3)


def test_declarative_layer_linter_passes_self_and_catches_banned_imports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert check_layer_imports(repo_root, raise_on_violation=True) == []
    assert layers_cli_main([str(repo_root)]) == 0

    # Ensure layers.py itself does not hardcode downstream package names outside docstrings
    layers_src = (repo_root / "src" / "basecoat" / "layers.py").read_text(encoding="utf-8")
    code_after_docstring = layers_src.split('"""', 2)[-1]
    for pkg in ("openlsrx", "multimodal_benchmark", "openimpala", "x_fuse", "braggtrack"):
        assert pkg not in code_after_docstring

    # Mock Layer 2 repo with banned import
    mock_repo = tmp_path / "mock_tool"
    (mock_repo / "src" / "mock_tool").mkdir(parents=True)
    toml_text = (
        '[project]\nname = "mock-tool"\n\n'
        "[tool.basecoat.layers]\n"
        "layer = 2\n"
        'source_dirs = ["src"]\n'
        'banned_imports = [\n  "forbidden_workflow",\n  "peer_solver",\n]\n'
    )
    (mock_repo / "pyproject.toml").write_text(toml_text, encoding="utf-8")
    parsed_fb = _parse_pyproject_layers_fallback(toml_text)
    assert parsed_fb["layer"] == 2
    assert parsed_fb["banned_imports"] == ["forbidden_workflow", "peer_solver"]

    bad_py = mock_repo / "src" / "mock_tool" / "bad.py"
    bad_py.write_text(
        "import numpy as np\nfrom forbidden_workflow.sub import helper\nimport peer_solver\n",
        encoding="utf-8",
    )
    violations = check_layer_imports(mock_repo)
    assert len(violations) == 2
    assert {v.banned_root for v in violations} == {"forbidden_workflow", "peer_solver"}
    with pytest.raises(BasecoatError, match="Layer 2 import boundary violated"):
        check_layer_imports(mock_repo, raise_on_violation=True)
    assert layers_cli_main([str(mock_repo)]) == 1
