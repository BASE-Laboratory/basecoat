"""Analytical tests for basecoat v0.3 conservative artefact filters and VolumeSet uncertainties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from basecoat import (
    SpaceMismatchError,
    Volume,
    VolumeSet,
    detect_ring_center_yx,
    detrend_radial_cupping,
    open_ome_zarr,
    read_ome_zarr,
    remove_rings_polar_conservative,
    write_ome_zarr,
)


def test_polar_ring_filter_preserves_sharp_edges_and_core_shell_chemistry_first() -> None:
    """Assert edge sharpness and radial chemistry retention FIRST, with ring suppression as floor."""
    rng = np.random.default_rng(7)
    ny, nx = 128, 128
    yy, xx = np.indices((ny, nx), dtype=np.float32)

    # Clean base microstructure: non-circular porous fragment + sharp small pores (3-5 px)
    clean = np.full((ny, nx), 60.0, dtype=np.float32)
    pellet = (((yy - 62.0) / 46.0) ** 2 + ((xx - 66.0) / 33.0) ** 2) <= 1.0
    clean[pellet] = 85.0
    for py, px, pr in [(50, 58, 3.2), (68, 74, 2.8), (58, 82, 3.5), (75, 55, 3.0), (45, 72, 2.6)]:
        clean[((yy - py) ** 2 + (xx - px) ** 2) <= pr**2] = 61.0
    clean = gaussian_filter(clean, sigma=0.7) + rng.normal(0.0, 0.4, size=(ny, nx)).astype(
        np.float32
    )

    # Inject concentric full detector rings + a 120-deg partial drift arc centred at (68.0, 52.0)
    cy_true, cx_true = 68.0, 52.0
    rr = np.hypot(yy - cy_true, xx - cx_true)
    th = np.arctan2(yy - cy_true, xx - cx_true)
    full_rings = 2.4 * np.exp(-0.5 * ((rr - 24.0) / 1.2) ** 2) - 2.1 * np.exp(
        -0.5 * ((rr - 39.0) / 1.4) ** 2
    )
    partial_arc = (
        2.6
        * np.exp(-0.5 * ((rr - 31.5) / 1.3) ** 2)
        * ((th > -0.8) & (th < 1.3)).astype(np.float32)
    )
    corrupted = clean + full_rings + partial_arc

    det_cy, det_cx, _ = detect_ring_center_yx(
        corrupted, search_y=(20, 110), search_x=(20, 110), radial_sigma_px=4.5
    )
    assert np.hypot(det_cy - cy_true, det_cx - cx_true) <= 2.5

    restored, rep = remove_rings_polar_conservative(
        corrupted,
        center_yx=(cy_true, cx_true),
        radial_sigma_px=4.5,
        angular_window_deg=45.0,
    )

    # 1. Edge sharpness retention FIRST (both reported & against ring-free ground-truth phantom)
    assert 0.98 <= rep.edge_sharpness_retention_ratio <= 1.02
    g_clean = np.hypot(
        gaussian_filter(clean, sigma=1.0, order=[1, 0]),
        gaussian_filter(clean, sigma=1.0, order=[0, 1]),
    )
    g_rest = np.hypot(
        gaussian_filter(restored, sigma=1.0, order=[1, 0]),
        gaussian_filter(restored, sigma=1.0, order=[0, 1]),
    )
    gt_edges = g_clean >= np.percentile(g_clean, 90.0)
    true_edge_ratio = float(np.mean(g_rest[gt_edges]) / np.mean(g_clean[gt_edges]))
    assert 0.98 <= true_edge_ratio <= 1.02

    # 2. Core-shell radial chemistry gradient retention FIRST (>92% slope retained)
    r_ell = np.sqrt(((yy - 62.0) / 46.0) ** 2 + ((xx - 66.0) / 33.0) ** 2)
    cs_chem = 6.0 * np.clip(r_ell, 0.0, 1.0) ** 2 * pellet.astype(np.float32)
    rest_cs, _ = remove_rings_polar_conservative(
        corrupted + cs_chem,
        center_yx=(cy_true, cx_true),
        radial_sigma_px=4.5,
        angular_window_deg=45.0,
    )
    recovered_cs = (rest_cs - restored)[pellet]
    slope_cs = float(np.polyfit(cs_chem[pellet], recovered_cs, 1)[0])
    assert slope_cs > 0.92

    # 3. Conservative ring suppression floor (including off-axis local-tomography crop cx < 0)
    err_before = float(np.sqrt(np.mean((corrupted - clean) ** 2)))
    err_after = float(np.sqrt(np.mean((restored - clean) ** 2)))
    assert err_after < 0.65 * err_before
    assert rep.ring_variance_reduction_pct > 40.0

    yy_off, xx_off = np.indices((96, 96), dtype=np.float32)
    rr_off = np.hypot(yy_off - 48.0, xx_off - (-30.0))
    cupping_off = 9.0 * (((yy_off - 48.0) ** 2 + (xx_off - 48.0) ** 2) / (48.0**2))
    off_vol = (
        np.full((4, 96, 96), 75.0, dtype=np.float32)
        + cupping_off[None, :, :]
        + (2.2 * np.sin(rr_off * 0.9) * np.exp(-0.5 * ((rr_off - 78.0) / 14.0) ** 2))[None, :, :]
    )
    _, rep_off = remove_rings_polar_conservative(
        off_vol, center_yx=(48.0, -30.0), radial_sigma_px=5.0
    )
    assert rep_off.ring_rms_removed > 0.25


def test_filters_preserve_volume_calibration_and_detrend_cupping() -> None:
    nz, ny, nx = 3, 96, 96
    yn = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[:, None]
    xn = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[None, :]
    r2 = yn * yn + xn * xn
    cup = 2.0 + 0.35 * r2 + 0.20 * (r2 * r2) - 0.08 * yn + 0.05 * xn
    arr = np.broadcast_to(cup[None, :, :], (nz, ny, nx)).copy()

    vol = Volume.from_array(
        arr,
        voxel_size_um=(1.6, 0.8, 0.8),
        origin_um=(10.0, 20.0, 30.0),
        space="i13_4x",
    )

    detrended = detrend_radial_cupping(vol)
    assert isinstance(detrended, Volume)
    assert detrended.axes == vol.axes
    assert detrended.space == "i13_4x"
    assert np.allclose(detrended.spacing_in("um"), (1.6, 0.8, 0.8))
    # Pure polynomial cupping collapses to flat mean within floating tolerance
    assert float(np.std(detrended.data)) < 1e-4
    assert np.isclose(float(np.mean(detrended.data)), float(np.mean(arr)), rtol=1e-4)

    derung, rep = remove_rings_polar_conservative(detrended, center_yx=(48.0, 48.0))
    assert isinstance(derung, Volume)
    assert derung.axes == vol.axes
    steps = [s["step"] for s in derung.provenance["processing_steps"]]
    assert steps == ["detrend_radial_cupping", "remove_rings_polar_conservative"]
    assert isinstance(rep.to_dict()["ring_center_yx"], tuple)


def test_volumeset_uncertainties_roundtrip_slicing_and_validation(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    img = Volume.from_array(
        rng.normal(size=(16, 16, 16)).astype(np.float32),
        voxel_size_um=0.8125,
        origin_um=(5.0, 10.0, 15.0),
        space="zn13x_roi",
    )
    phase = Volume.from_array(
        (img.data > 0.0).astype(np.uint8),
        voxel_size_um=0.8125,
        origin_um=(5.0, 10.0, 15.0),
        space="zn13x_roi",
        label_names={0: "pore", 1: "solid"},
    )
    sigma_ortho = Volume.from_array(
        np.abs(rng.normal(scale=0.05, size=(16, 16, 16))).astype(np.float32),
        voxel_size_um=0.8125,
        origin_um=(5.0, 10.0, 15.0),
        space="zn13x_roi",
    )
    probs_raw = rng.uniform(0.1, 0.9, size=(3, 16, 16, 16)).astype(np.float32)
    probs_raw /= np.sum(probs_raw, axis=0, keepdims=True)
    class_probs = Volume.from_array(
        probs_raw,
        voxel_size_um=0.8125,
        origin_um=(5.0, 10.0, 15.0),
        axes_names=("c", "z", "y", "x"),
        space="zn13x_roi",
    )

    # Validation: integer uncertainty must raise ValueError
    with pytest.raises(ValueError, match="floating-point dtype"):
        VolumeSet(img, uncertainties={"bad_int": phase})

    # Validation: mismatched scale must raise SpaceMismatchError
    wrong_scale = Volume.from_array(
        sigma_ortho.data, voxel_size_um=1.0, origin_um=(5.0, 10.0, 15.0), space="zn13x_roi"
    )
    with pytest.raises(SpaceMismatchError):
        VolumeSet(img, uncertainties={"sigma_ortho": wrong_scale})

    vset = VolumeSet(
        img,
        labels={"phase": phase},
        uncertainties={"sigma_ortho": sigma_ortho, "class_probs": class_probs},
    )
    assert sorted(vset.uncertainties.keys()) == ["class_probs", "sigma_ortho"]

    # Coherent slicing across 3D image, 3D label, 3D scalar uncertainty, and 4D channel probabilities
    sub = vset[4:12, 2:10, 6:14]
    assert sub.image.shape == (8, 8, 8)
    assert sub.labels["phase"].shape == (8, 8, 8)
    assert sub.uncertainties["sigma_ortho"].shape == (8, 8, 8)
    assert sub.uncertainties["class_probs"].shape == (3, 8, 8, 8)
    assert sub.uncertainties["sigma_ortho"].origin_in("um") == sub.image.origin_in("um")
    assert sub.uncertainties["class_probs"].origin_in("um") == sub.image.origin_in("um")

    # Multiscale OME-Zarr v0.4 round-trip
    store_path = write_ome_zarr(vset, tmp_path / "with_unc.ome.zarr", multiscale_levels=2)

    rt0 = read_ome_zarr(store_path, level=0, as_set=True)
    assert np.allclose(rt0.uncertainties["sigma_ortho"].data, sigma_ortho.data)
    assert np.allclose(rt0.uncertainties["class_probs"].data, class_probs.data)

    rt1 = read_ome_zarr(store_path, level=1, as_set=True)
    assert rt1.uncertainties["sigma_ortho"].shape == (8, 8, 8)
    assert rt1.uncertainties["class_probs"].shape == (3, 8, 8, 8)

    # Lazy uncertainty region read
    with open_ome_zarr(store_path, uncertainty="sigma_ortho") as lv:
        roi = lv.read_region((slice(0, 4), slice(0, 4), slice(0, 4)))
        assert np.allclose(roi.data, sigma_ortho.data[:4, :4, :4])

    # Community ome-zarr reader still validates root + labels cleanly
    ome_io = pytest.importorskip("ome_zarr.io")
    ome_reader = pytest.importorskip("ome_zarr.reader")
    loc = ome_io.parse_url(str(store_path))
    assert loc is not None
    nodes = list(ome_reader.Reader(loc)())
    assert len(nodes) >= 2
