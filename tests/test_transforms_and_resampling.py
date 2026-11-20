"""Analytical tests for cell-centre geometry, pullback transforms, and MultimodalBenchmark SimpleITK parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from basecoat import (
    AffineTransform3D,
    AxisSpec,
    SimilarityTransform3D,
    SpaceMismatchError,
    Volume,
    coarsen_axes,
    resample_volume,
)


def test_cell_centre_resampling_analytical_linear_ramp() -> None:
    """Ties AxisSpec cell-centre definition (x(i) = offset + i * scale) directly to resample_volume."""
    nz, ny, nx = 24, 28, 32
    sz, sy, sx = 0.8125, 0.8125, 0.8125
    oz, oy, ox = -12.0, 8.5, 3.25

    fine_axes = (
        AxisSpec("z", "space", "um", sz, oz),
        AxisSpec("y", "space", "um", sy, oy),
        AxisSpec("x", "space", "um", sx, ox),
    )

    zz = fine_axes[0].index_to_world(np.arange(nz))[:, None, None]
    yy = fine_axes[1].index_to_world(np.arange(ny))[None, :, None]
    xx = fine_axes[2].index_to_world(np.arange(nx))[None, None, :]

    # Exact linear field f(z, y, x) = 1.7*z - 0.9*y + 2.3*x + 4.5
    field_fine = 1.7 * zz - 0.9 * yy + 2.3 * xx + 4.5
    vol_fine = Volume(field_fine, fine_axes, space="pellet_um")

    # 1. Resample onto 2x coarsened grid (offset_coarse = offset + 0.5 * scale)
    coarse_axes = coarsen_axes(fine_axes, factors=2)
    resampled_coarse = resample_volume(
        vol_fine,
        coarse_axes,
        target_shape=(nz // 2, ny // 2, nx // 2),
    )

    zc = coarse_axes[0].index_to_world(np.arange(nz // 2))[:, None, None]
    yc = coarse_axes[1].index_to_world(np.arange(ny // 2))[None, :, None]
    xc = coarse_axes[2].index_to_world(np.arange(nx // 2))[None, None, :]
    expected_coarse = 1.7 * zc - 0.9 * yc + 2.3 * xc + 4.5

    # Every interior voxel (where [i_in] lies inside [0, N-1]) matches to float64 precision!
    assert (
        np.max(np.abs(resampled_coarse.data[1:-1, 1:-1, 1:-1] - expected_coarse[1:-1, 1:-1, 1:-1]))
        < 1e-11
    )

    # 2. Resample onto an arbitrary non-integer spacing grid inside the interior domain
    arb_axes = (
        AxisSpec("z", "space", "um", 1.37, oz + 2.0),
        AxisSpec("y", "space", "um", 1.19, oy + 2.0),
        AxisSpec("x", "space", "um", 1.53, ox + 2.0),
    )
    resampled_arb = resample_volume(vol_fine, arb_axes, target_shape=(10, 12, 12))
    za = arb_axes[0].index_to_world(np.arange(10))[:, None, None]
    ya = arb_axes[1].index_to_world(np.arange(12))[None, :, None]
    xa = arb_axes[2].index_to_world(np.arange(12))[None, None, :]
    expected_arb = 1.7 * za - 0.9 * ya + 2.3 * xa + 4.5
    assert np.max(np.abs(resampled_arb.data - expected_arb)) < 1e-11


def test_transform_chaining_inversion_and_safety_guards(tmp_path: Path) -> None:
    sim_ab = SimilarityTransform3D.from_euler_xyz(
        euler_xyz_rad=(0.05, -0.08, 0.12),
        scale=1.6039,
        translation_xyz=(-1.2, 2.4, 0.8),
        center_xyz=(50.0, 60.0, 70.0),
        from_space="rung_b",
        to_space="rung_a",
    )
    sim_ba = sim_ab.inverse()
    pts = np.array([[10.0, 20.0, 30.0], [85.0, 42.0, 19.5]], dtype=np.float64)
    round_trip = sim_ba.transform_points(sim_ab.transform_points(pts))
    assert np.allclose(round_trip, pts, atol=1e-11)

    # JSON round-trip
    json_path = sim_ab.save_json(tmp_path / "sim.json")
    reloaded = SimilarityTransform3D.load_json(json_path)
    assert np.allclose(reloaded.transform_points(pts), sim_ab.transform_points(pts), atol=1e-12)

    # Chaining space check
    aff_bc = AffineTransform3D.from_linear_and_offset(
        np.diag([2.0, 2.0, 2.0]),
        [1.0, -2.0, 3.0],
        from_space="rung_c",
        to_space="rung_b",
    )
    chained_ca = aff_bc.then(sim_ab.to_affine())
    assert chained_ca.from_space == "rung_c"
    assert chained_ca.to_space == "rung_a"
    assert np.allclose(
        chained_ca.transform_points(pts),
        sim_ab.transform_points(aff_bc.transform_points(pts)),
        atol=1e-11,
    )

    with pytest.raises(SpaceMismatchError):
        sim_ab.to_affine().then(aff_bc)  # Backwards: rung_a != rung_c

    # Guard against accidentally passing SimpleITK's already-scaled GetMatrix() into rotation_zyx
    with pytest.raises(ValueError, match="orthonormal"):
        SimilarityTransform3D(
            rotation_zyx=2.0 * np.eye(3),
            scale=2.0,
            center_zyx=(0.0, 0.0, 0.0),
            translation_zyx=(0.0, 0.0, 0.0),
            from_space="a",
            to_space="b",
        )


def test_reproduce_multimodal_benchmark_2p026x_sitk_resample() -> None:
    """Empirical regression check against MultimodalBenchmark's SimpleITK 2p026x resampling pipeline."""
    sitk = pytest.importorskip("SimpleITK")

    # Exact 2p026x parameters from MultimodalBenchmark configs/i13_benchmark_transforms.json
    lr_nom_um = 1.625
    hr_nom_um = 0.8125
    hr_shape_zyx = (40, 44, 48)
    lr_shape_zyx = (24, 26, 28)

    euler_xyz_rad = [-0.00112, -0.00095, -0.00063]
    fitted_scale = 0.98727
    fine_translation_xyz_um = [-5.284, -1.498, 2.345]

    cx = 0.5 * (hr_shape_zyx[2] - 1) * hr_nom_um
    cy = 0.5 * (hr_shape_zyx[1] - 1) * hr_nom_um
    cz = 0.5 * (hr_shape_zyx[0] - 1) * hr_nom_um

    # Smooth + localized porous phantom on LR grid
    zz, yy, xx = np.meshgrid(
        np.arange(lr_shape_zyx[0], dtype=np.float32) * lr_nom_um,
        np.arange(lr_shape_zyx[1], dtype=np.float32) * lr_nom_um,
        np.arange(lr_shape_zyx[2], dtype=np.float32) * lr_nom_um,
        indexing="ij",
    )
    lr_data = (
        np.sin(0.08 * zz) * np.cos(0.07 * yy)
        + 0.6 * np.exp(-((zz - 18.0) ** 2 + (yy - 20.0) ** 2 + (xx - 22.0) ** 2) / 45.0)
        + 0.02 * xx
    ).astype(np.float32)
    fill_val = float(np.percentile(lr_data, 5.0))

    # 1. Reference MultimodalBenchmark SimpleITK pipeline
    s3d = sitk.Similarity3DTransform()
    s3d.SetCenter([cx, cy, cz])
    e3d = sitk.Euler3DTransform()
    e3d.SetRotation(*euler_xyz_rad)
    s3d.SetMatrix(e3d.GetMatrix())
    s3d.SetScale(fitted_scale)
    s3d.SetTranslation(fine_translation_xyz_um)

    mov_img = sitk.GetImageFromArray(lr_data)
    mov_img.SetSpacing((lr_nom_um, lr_nom_um, lr_nom_um))
    mov_img.SetOrigin((0.0, 0.0, 0.0))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((hr_shape_zyx[2], hr_shape_zyx[1], hr_shape_zyx[0]))
    resampler.SetOutputSpacing((hr_nom_um, hr_nom_um, hr_nom_um))
    resampler.SetOutputOrigin((0.0, 0.0, 0.0))
    resampler.SetTransform(s3d)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(fill_val)
    sitk_out = sitk.GetArrayFromImage(resampler.Execute(mov_img)).astype(np.float32)

    # 2. basecoat via .from_sitk() AND pure-Python .from_euler_xyz() (no SimpleITK needed at runtime)
    lr_vol = Volume.from_array(lr_data, voxel_size_um=lr_nom_um, space="i13_169067")
    hr_grid_axes = (
        AxisSpec("z", "space", "um", hr_nom_um, 0.0),
        AxisSpec("y", "space", "um", hr_nom_um, 0.0),
        AxisSpec("x", "space", "um", hr_nom_um, 0.0),
    )

    tfm_from_sitk = SimilarityTransform3D.from_sitk(
        s3d, from_space="i13_169066", to_space="i13_169067"
    )
    tfm_pure_euler = SimilarityTransform3D.from_euler_xyz(
        euler_xyz_rad=euler_xyz_rad,
        scale=fitted_scale,
        translation_xyz=fine_translation_xyz_um,
        center_xyz=(cx, cy, cz),
        from_space="i13_169066",
        to_space="i13_169067",
    )

    bc_from_sitk = resample_volume(
        lr_vol,
        hr_grid_axes,
        target_shape=hr_shape_zyx,
        transform=tfm_from_sitk,
        order=1,
        cval=fill_val,
    )
    bc_pure = resample_volume(
        lr_vol,
        hr_grid_axes,
        target_shape=hr_shape_zyx,
        transform=tfm_pure_euler,
        order=1,
        cval=fill_val,
    )

    assert np.allclose(bc_from_sitk.data, bc_pure.data, atol=1e-7)
    # Given tx = -5.284 um and hr_nom = 0.8125 um, mapped x_lr >= 0 starts at ix >= 7.
    # Every voxel mapped inside [0, N_lr - 1] matches SimpleITK trilinear interpolation to float32 precision.
    interior = (slice(4, 34), slice(4, 38), slice(8, 42))
    max_diff = float(np.max(np.abs(bc_pure.data[interior] - sitk_out[interior])))
    assert max_diff < 1e-5, f"Max interior difference vs SimpleITK was {max_diff:.3e}"
