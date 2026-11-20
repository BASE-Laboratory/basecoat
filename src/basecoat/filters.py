"""Conservative ("do-no-harm") tomography artefact filters (`polar_ring` & `radial_cupping`).

Design guarantees (`ECOSYSTEM.md` §Hard rules):
  - Zero dataset-specific scan IDs, dimensions, or thresholds.
  - High-contrast structural boundaries (pellet contours, capillary walls, calibration spheres,
    macropore walls) are Huber-clamped during polar ring estimation and protected by a smooth
    gradient weight so real phase boundaries never induce false counter-arcs and edge sharpness
    retention stays within ~99-101%.
  - Radial cupping detrending explicitly documents core-shell chemistry confounding so callers
    never erase genuine radial chemical attenuation gradients on reference volumes.

Extracted under BSD-3-Clause from ``MultimodalBenchmark/preprocessing/artifact_cleanup.py``,
crediting James Le Houx and Calum Green.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, overload

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    map_coordinates,
    median_filter,
)

from ._provenance import stamp_provenance
from .volume import Volume


@dataclass(frozen=True)
class ArtifactCleanupReport:
    """Quantitative safety and effectiveness metrics for conservative polar ring cleanup."""

    ring_center_yx: tuple[float, float]
    ring_rms_removed: float
    ring_anova_r2_before: float
    ring_anova_r2_after: float
    ring_variance_reduction_pct: float
    edge_sharpness_retention_ratio: float
    background_std_before: float
    background_std_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_ring_anova_r2(
    img_yx: np.ndarray[Any, Any],
    center_yx: tuple[float, float],
    *,
    exclude_mask_yx: np.ndarray[Any, Any] | None = None,
    radial_sigma_px: float = 5.0,
    mad_clip: float = 1.8,
) -> float:
    """Return adjusted one-way ANOVA fraction of bandpass variance explained by concentric rings."""
    sl = np.asarray(img_yx, dtype=np.float32)
    if sl.ndim != 2:
        raise ValueError(f"compute_ring_anova_r2 expects a 2D (y, x) slice, got shape={sl.shape}.")

    bp = gaussian_filter(sl, sigma=0.8) - gaussian_filter(sl, sigma=float(radial_sigma_px))
    gy = gaussian_filter(sl, sigma=1.5, order=[1, 0])
    gx = gaussian_filter(sl, sigma=1.5, order=[0, 1])
    gmag = np.hypot(gy, gx)
    strong_edge = np.asarray(
        binary_dilation(
            gmag > np.percentile(gmag, 85.0),
            iterations=max(2, int(round(radial_sigma_px))),
        ),
        dtype=bool,
    )

    valid = (
        (~strong_edge)
        if exclude_mask_yx is None
        else ((~strong_edge) & (~np.asarray(exclude_mask_yx, dtype=bool)))
    )
    if int(np.count_nonzero(valid)) < 256:
        valid = np.ones(sl.shape, dtype=bool)

    mad = float(np.median(np.abs(bp[valid] - np.median(bp[valid])))) + 1e-6
    hp = np.clip(bp, -mad_clip * mad, mad_clip * mad)

    ny, nx = sl.shape
    yy, xx = np.indices((ny, nx), dtype=np.float32)

    ys = yy[valid] - np.float32(center_yx[0])
    xs = xx[valid] - np.float32(center_yx[1])
    vals = hp[valid].astype(np.float64)
    vals -= float(np.mean(vals))
    vvar = float(np.var(vals))
    denom = float(np.dot(vals, vals)) + 1e-12

    rr = np.rint(np.hypot(ys, xs)).astype(np.int32)
    max_r = int(rr.max()) + 1
    cnt = np.bincount(rr, minlength=max_r).astype(np.float64)
    ok = cnt >= 16
    if int(np.count_nonzero(ok)) < 12:
        return 0.0
    sums = np.bincount(rr, weights=vals, minlength=max_r)
    adj_num = float(np.sum((sums[ok] ** 2) / cnt[ok] - vvar))
    return max(0.0, adj_num / denom)


def detect_ring_center_yx(
    vol_or_slice: np.ndarray[Any, Any] | Volume,
    *,
    exclude_mask_yx: np.ndarray[Any, Any] | None = None,
    search_y: tuple[int, int] | None = None,
    search_x: tuple[int, int] | None = None,
    radial_sigma_px: float = 5.0,
) -> tuple[float, float, float]:
    """Locate the concentric rotation/ring centre ``(cy, cx, adjusted_R2)``, including off-axis crops."""
    raw = vol_or_slice.data if isinstance(vol_or_slice, Volume) else np.asarray(vol_or_slice)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"detect_ring_center_yx expects a 2D (y, x) or 3D (z, y, x) array, got shape={arr.shape}."
        )
    zmean = np.mean(arr, axis=0) if arr.ndim == 3 else arr
    ny, nx = zmean.shape

    sy = search_y if search_y is not None else (-ny // 2, int(round(1.5 * ny)))
    sx = search_x if search_x is not None else (-nx // 2, int(round(1.5 * nx)))

    bp = gaussian_filter(zmean, sigma=0.8) - gaussian_filter(zmean, sigma=float(radial_sigma_px))
    gy = gaussian_filter(zmean, sigma=1.5, order=[1, 0])
    gx = gaussian_filter(zmean, sigma=1.5, order=[0, 1])
    gmag = np.hypot(gy, gx)
    strong_edge = np.asarray(
        binary_dilation(
            gmag > np.percentile(gmag, 85.0),
            iterations=max(4, int(round(2.2 * radial_sigma_px))),
        ),
        dtype=bool,
    )
    excl = (
        strong_edge
        if exclude_mask_yx is None
        else (strong_edge | np.asarray(exclude_mask_yx, dtype=bool))
    )
    valid_px = ~excl
    if int(np.count_nonzero(valid_px)) < 256:
        valid_px = np.ones(zmean.shape, dtype=bool)

    mad = float(np.median(np.abs(bp[valid_px] - np.median(bp[valid_px])))) + 1e-6
    hp = np.clip(bp, -1.8 * mad, 1.8 * mad)
    hp[~valid_px] = 0.0

    yy, xx = np.indices((ny, nx), dtype=np.float32)
    grid_step_coarse = max(4, min(32, min(ny, nx) // 16))
    img_step_coarse = max(1, min(4, min(ny, nx) // 64))

    def _score(cy: float, cx: float, step: int) -> float:
        m = valid_px[::step, ::step].ravel()
        if int(np.count_nonzero(m)) < 128:
            return -1.0
        ys = yy[::step, ::step].ravel()[m] - np.float32(cy)
        xs = xx[::step, ::step].ravel()[m] - np.float32(cx)
        vals = hp[::step, ::step].ravel()[m].astype(np.float64)
        vals -= float(np.mean(vals))
        vvar = float(np.var(vals))
        denom = float(np.dot(vals, vals)) + 1e-12
        rr = np.rint(np.hypot(ys, xs) / step).astype(np.int32)
        max_r = int(rr.max()) + 1
        cnt = np.bincount(rr, minlength=max_r).astype(np.float64)
        ok = cnt >= max(12, 24 // step)
        if int(np.count_nonzero(ok)) < 12:
            return -1.0
        sums = np.bincount(rr, weights=vals, minlength=max_r)
        return float(np.sum((sums[ok] ** 2) / cnt[ok] - vvar) / denom)

    best = (-2.0, 0.5 * (ny - 1), 0.5 * (nx - 1))
    for cy in range(int(sy[0]), int(sy[1]) + 1, grid_step_coarse):
        for cx in range(int(sx[0]), int(sx[1]) + 1, grid_step_coarse):
            s = _score(float(cy), float(cx), step=img_step_coarse)
            if s > best[0]:
                best = (s, float(cy), float(cx))

    _, bcy, bcx = best
    bloc = best
    for sg, si, hw in [
        (max(2.0, grid_step_coarse / 4.0), max(1, img_step_coarse // 2), float(grid_step_coarse)),
        (1.0, 1, max(4.0, grid_step_coarse / 2.0)),
        (0.5, 1, 2.0),
    ]:
        bloc = (-2.0, bcy, bcx)
        for cy_f in np.arange(bcy - hw, bcy + hw + 0.05, sg):
            for cx_f in np.arange(bcx - hw, bcx + hw + 0.05, sg):
                s = _score(float(cy_f), float(cx_f), step=si)
                if s > bloc[0]:
                    bloc = (s, float(cy_f), float(cx_f))
        _, bcy, bcx = bloc

    return float(bcy), float(bcx), max(0.0, float(bloc[0]))


def _estimate_single_slice_ring_map(
    sl_yx: np.ndarray[Any, Any],
    center_yx: tuple[float, float],
    *,
    radial_sigma_px: float = 5.5,
    angular_window_deg: float = 50.0,
    mad_clip_factor: float = 2.5,
    max_correction_sigma: float = 0.35,
    protect_strong_edges: bool = True,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Estimate halo-free conservative ring/arc artefact field via coupled polar median filtering."""
    ny, nx = sl_yx.shape
    cy, cx = float(center_yx[0]), float(center_yx[1])

    gy = gaussian_filter(sl_yx, sigma=1.0, order=[1, 0])
    gx = gaussian_filter(sl_yx, sigma=1.0, order=[0, 1])
    gmag = np.hypot(gy, gx)
    p10, p95 = (float(v) for v in np.percentile(sl_yx, [10.0, 95.0]))
    contrast_range = max(p95 - p10, float(np.std(sl_yx)) * 2.0, 1e-5)
    g_edge_thresh = 0.32 * contrast_range
    g_p50 = float(np.percentile(gmag, 50.0))

    yy, xx = np.indices((ny, nx), dtype=np.float32)
    dy = yy - np.float32(cy)
    dx = xx - np.float32(cx)
    r_map = np.hypot(dy, dx)
    th_map = np.arctan2(dy, dx)

    r_min = max(0.0, float(np.floor(r_map.min())) - 1.0)
    r_max = float(np.ceil(r_map.max())) + 1.0
    nr = max(12, int(round(r_max - r_min)) + 1)
    r_grid = np.linspace(r_min, r_max, nr, dtype=np.float32)

    center_inside = (0.0 <= cy <= ny - 1) and (0.0 <= cx <= nx - 1)
    r_med = max(60.0, float(np.median(r_map)))
    dtheta = float(np.clip(1.25 / r_med, np.deg2rad(0.25), np.deg2rad(1.0)))

    if center_inside:
        ntheta = max(64, int(round(2.0 * np.pi / dtheta)))
        th_grid = np.linspace(-np.pi, np.pi, ntheta, endpoint=False, dtype=np.float32)
        ang_mode = "wrap"
        th_min, th_span = -np.pi, 2.0 * np.pi
    else:
        th_ref = float(np.arctan2(0.5 * ny - cy, 0.5 * nx - cx))
        th_rel = ((th_map - th_ref + np.pi) % (2.0 * np.pi)) - np.pi
        t0 = float(th_rel.min()) - 2.0 * dtheta
        t1 = float(th_rel.max()) + 2.0 * dtheta
        ntheta = max(32, int(round((t1 - t0) / dtheta)) + 1)
        th_grid_rel = np.linspace(t0, t1, ntheta, dtype=np.float32)
        th_grid = th_ref + th_grid_rel
        th_map = th_rel
        th_min, th_span = t0, (t1 - t0)
        ang_mode = "reflect"

    py = cy + r_grid[:, None] * np.sin(th_grid)[None, :]
    px = cx + r_grid[:, None] * np.cos(th_grid)[None, :]
    polar_img = map_coordinates(sl_yx, [py, px], order=3, mode="nearest")

    win_r = max(7, int(round(2.4 * float(radial_sigma_px))) | 1)
    win_r = min(win_r, max(7, (nr // 2) | 1))
    polar_hp = polar_img - median_filter(polar_img, size=(win_r, 1), mode="nearest")

    if protect_strong_edges:
        wall_thresh = max(0.23 * contrast_range, float(np.percentile(gmag, 95.0)), 4.0 * g_p50)
        wall_yx = gmag > wall_thresh
        wall_polar = (
            map_coordinates(wall_yx.astype(np.float32), [py, px], order=1, mode="nearest") > 0.12
        )
        invalid_polar = np.asarray(
            binary_dilation(wall_polar, structure=np.ones((5, 3), dtype=bool)), dtype=bool
        )
    else:
        invalid_polar = np.zeros_like(polar_hp, dtype=bool)

    valid_vals = polar_hp[~invalid_polar]
    if valid_vals.size < 64:
        valid_vals = polar_hp.ravel()
    mad = float(np.median(np.abs(valid_vals - np.median(valid_vals)))) + 1e-6
    robust_sig = 1.4826 * mad
    tau_ring = max(float(mad_clip_factor) * robust_sig, 0.14 * contrast_range)
    polar_clamped = np.clip(polar_hp, -tau_ring, tau_ring).astype(np.float32)

    if np.any(invalid_polar):
        min_valid = max(6, ntheta // 12)
        for ir_row in np.flatnonzero(np.any(invalid_polar, axis=1)):
            v_idx = np.flatnonzero(~invalid_polar[ir_row])
            if v_idx.size < min_valid:
                polar_clamped[ir_row, :] = 0.0
                continue
            inv_idx = np.flatnonzero(invalid_polar[ir_row])
            v_fp = polar_clamped[ir_row, v_idx]
            if center_inside:
                xp = np.concatenate([v_idx - ntheta, v_idx, v_idx + ntheta])
                fp = np.tile(v_fp, 3)
                polar_clamped[ir_row, inv_idx] = np.interp(inv_idx, xp, fp).astype(np.float32)
            else:
                polar_clamped[ir_row, inv_idx] = np.interp(inv_idx, v_idx, v_fp).astype(np.float32)

    win_th = max(5, int(round(np.deg2rad(angular_window_deg) / max(dtheta, 1e-4))) | 1)
    win_th = min(win_th, max(5, (ntheta // 2) | 1))
    polar_ring = median_filter(polar_clamped, size=(1, win_th), mode=ang_mode)
    polar_ring = polar_ring - median_filter(polar_ring, size=(win_r, 1), mode="nearest")

    ir = (r_map - r_min) / max(r_max - r_min, 1e-6) * (nr - 1)
    if center_inside:
        ith = ((th_map - th_min) % (2.0 * np.pi)) / (2.0 * np.pi) * ntheta
        ring_yx = map_coordinates(polar_ring, [ir, ith], order=3, mode="wrap")
    else:
        ith = (th_map - th_min) / max(th_span, 1e-6) * (ntheta - 1)
        ring_yx = map_coordinates(polar_ring, [ir, ith], order=3, mode="nearest")

    if protect_strong_edges:
        edge_prot_out = np.exp(
            -0.5 * (np.maximum(0.0, gmag - g_edge_thresh) / (0.60 * g_edge_thresh + 1e-6)) ** 2
        ).astype(np.float32)
        ring_yx = ring_yx * edge_prot_out

    core_damp = np.clip((r_map - 2.0) / 5.0, 0.25, 1.0).astype(np.float32)
    ring_yx = ring_yx * core_damp

    max_corr = float(max_correction_sigma) * float(np.std(sl_yx) + 1e-6)
    ring_yx = np.clip(ring_yx, -max_corr, max_corr).astype(np.float32)
    low_grad_mask = gmag <= g_p50
    return ring_yx, gmag, low_grad_mask


@overload
def remove_rings_polar_conservative(
    vol_or_slice: Volume,
    center_yx: tuple[float, float] | None = None,
    *,
    radial_sigma_px: float = 5.5,
    angular_window_deg: float = 50.0,
    mad_clip_factor: float = 2.2,
    max_correction_sigma: float = 0.35,
    protect_strong_edges: bool = True,
    exclude_mask_yx: np.ndarray[Any, Any] | None = None,
) -> tuple[Volume, ArtifactCleanupReport]: ...


@overload
def remove_rings_polar_conservative(
    vol_or_slice: np.ndarray[Any, Any],
    center_yx: tuple[float, float] | None = None,
    *,
    radial_sigma_px: float = 5.5,
    angular_window_deg: float = 50.0,
    mad_clip_factor: float = 2.2,
    max_correction_sigma: float = 0.35,
    protect_strong_edges: bool = True,
    exclude_mask_yx: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any], ArtifactCleanupReport]: ...


def remove_rings_polar_conservative(
    vol_or_slice: np.ndarray[Any, Any] | Volume,
    center_yx: tuple[float, float] | None = None,
    *,
    radial_sigma_px: float = 5.5,
    angular_window_deg: float = 50.0,
    mad_clip_factor: float = 2.2,
    max_correction_sigma: float = 0.35,
    protect_strong_edges: bool = True,
    exclude_mask_yx: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any] | Volume, ArtifactCleanupReport]:
    """Conservatively remove full and partial-arc ring artefacts from a 2D slice, 3D array, or :class:`Volume`."""
    is_volume = isinstance(vol_or_slice, Volume)
    raw = vol_or_slice.data if is_volume else np.asarray(vol_or_slice)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"remove_rings_polar_conservative expects 2D (y, x) or 3D (z, y, x), got shape={arr.shape}."
        )
    is_2d = arr.ndim == 2
    vol = arr[None, :, :] if is_2d else arr

    if center_yx is None:
        cy, cx, _ = detect_ring_center_yx(
            vol[: min(12, vol.shape[0])],
            exclude_mask_yx=exclude_mask_yx,
            radial_sigma_px=radial_sigma_px,
        )
        center_yx = (cy, cx)

    cleaned = np.empty_like(vol, dtype=np.float32)
    ring_sq_sum = 0.0
    edge_ratios: list[float] = []
    bg_stds_before: list[float] = []
    bg_stds_after: list[float] = []

    for iz in range(vol.shape[0]):
        sl = vol[iz]
        ring_yx, g_before, low_g = _estimate_single_slice_ring_map(
            sl,
            center_yx=center_yx,
            radial_sigma_px=radial_sigma_px,
            angular_window_deg=angular_window_deg,
            mad_clip_factor=mad_clip_factor,
            max_correction_sigma=max_correction_sigma,
            protect_strong_edges=protect_strong_edges,
        )
        cl = sl - ring_yx
        cleaned[iz] = cl
        ring_sq_sum += float(np.mean(ring_yx * ring_yx))

        gy_a = gaussian_filter(cl, sigma=1.0, order=[1, 0])
        gx_a = gaussian_filter(cl, sigma=1.0, order=[0, 1])
        g_after = np.hypot(gy_a, gx_a)
        top_e = g_before >= np.percentile(g_before, 90.0)
        edge_ratios.append(float(np.mean(g_after[top_e]) / (np.mean(g_before[top_e]) + 1e-8)))

        if np.any(low_g):
            hp_b = sl - gaussian_filter(sl, sigma=radial_sigma_px)
            hp_a = cl - gaussian_filter(cl, sigma=radial_sigma_px)
            bg_stds_before.append(float(np.std(hp_b[low_g])))
            bg_stds_after.append(float(np.std(hp_a[low_g])))

    mid_b = np.mean(vol[: min(10, vol.shape[0])], axis=0)
    mid_a = np.mean(cleaned[: min(10, cleaned.shape[0])], axis=0)
    r2_b = compute_ring_anova_r2(
        mid_b, center_yx, exclude_mask_yx=exclude_mask_yx, radial_sigma_px=radial_sigma_px
    )
    r2_a = compute_ring_anova_r2(
        mid_a, center_yx, exclude_mask_yx=exclude_mask_yx, radial_sigma_px=radial_sigma_px
    )
    red_pct = (1.0 - r2_a / max(r2_b, 1e-8)) * 100.0 if r2_b > 1e-6 else 0.0

    report = ArtifactCleanupReport(
        ring_center_yx=(round(float(center_yx[0]), 2), round(float(center_yx[1]), 2)),
        ring_rms_removed=round(float(np.sqrt(ring_sq_sum / vol.shape[0])), 4),
        ring_anova_r2_before=round(r2_b, 6),
        ring_anova_r2_after=round(r2_a, 6),
        ring_variance_reduction_pct=round(float(red_pct), 2),
        edge_sharpness_retention_ratio=round(float(np.mean(edge_ratios)), 4),
        background_std_before=round(float(np.mean(bg_stds_before)) if bg_stds_before else 0.0, 4),
        background_std_after=round(float(np.mean(bg_stds_after)) if bg_stds_after else 0.0, 4),
    )
    out_arr = cleaned[0] if is_2d else cleaned
    if isinstance(vol_or_slice, Volume):
        prov = stamp_provenance(vol_or_slice.provenance)
        steps = list(prov.get("processing_steps", []))
        steps.append({"step": "remove_rings_polar_conservative", "report": report.to_dict()})
        prov["processing_steps"] = steps
        return (
            Volume(
                out_arr,
                vol_or_slice.axes,
                space=vol_or_slice.space,
                centering=vol_or_slice.centering,
                provenance=prov,
            ),
            report,
        )
    return out_arr, report


@overload
def detrend_radial_cupping(
    vol_zyx: Volume,
    *,
    sigma_yx_vox: float = 48.0,
    valid_mask_zyx: np.ndarray[Any, Any] | None = None,
) -> Volume: ...


@overload
def detrend_radial_cupping(
    vol_zyx: np.ndarray[Any, Any],
    *,
    sigma_yx_vox: float = 48.0,
    valid_mask_zyx: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]: ...


def detrend_radial_cupping(
    vol_zyx: np.ndarray[Any, Any] | Volume,
    *,
    sigma_yx_vox: float = 48.0,
    valid_mask_zyx: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any] | Volume:
    """Remove low-order 2D quadratic + radial-quartic ``[1, y, x, y^2, y*x, x^2, r^4]`` background cupping.

    .. warning::
       **Chemistry Confounding:** In samples with radial chemical composition gradients (e.g.
       core-shell ion-exchanged pellets), a 2D radial polynomial fit correlates with genuine
       photoelectric attenuation gradients. Restrict this filter to chemically homogeneous
       carriers or diagnostic previewing; never overwrite quantitative reference intensities
       on chemically graded specimens.
    """
    _ = sigma_yx_vox
    is_volume = isinstance(vol_zyx, Volume)
    raw = vol_zyx.data if is_volume else np.asarray(vol_zyx)
    vol = np.asarray(raw, dtype=np.float32)
    if vol.ndim not in (2, 3):
        raise ValueError(
            f"detrend_radial_cupping expects 2D (y, x) or 3D (z, y, x), got shape={vol.shape}."
        )
    is_2d = vol.ndim == 2
    v3 = vol[None, :, :] if is_2d else vol
    nz, ny, nx = v3.shape

    yn = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[:, None]
    xn = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[None, :]
    yy = np.broadcast_to(yn, (ny, nx))
    xx = np.broadcast_to(xn, (ny, nx))
    r2 = yy * yy + xx * xx
    basis_full = np.stack([np.ones_like(yy), yy, xx, yy * yy, yy * xx, xx * xx, r2 * r2], axis=-1)

    stride = max(1, min(ny, nx) // 128)
    b_sub = basis_full[::stride, ::stride].reshape(-1, 7).astype(np.float64)
    out = np.empty_like(v3, dtype=np.float32)

    for iz in range(nz):
        sl = v3[iz]
        sl_sub = sl[::stride, ::stride].ravel().astype(np.float64)
        if valid_mask_zyx is not None:
            m_sl = valid_mask_zyx if valid_mask_zyx.ndim == 2 else valid_mask_zyx[iz]
            m_sub = (m_sl[::stride, ::stride] > 0).ravel()
        else:
            m_sub = np.ones_like(sl_sub, dtype=bool)
        if int(np.count_nonzero(m_sub)) < 32:
            m_sub = np.ones_like(sl_sub, dtype=bool)

        coeffs, _, _, _ = np.linalg.lstsq(b_sub[m_sub], sl_sub[m_sub], rcond=None)
        cup_surface = np.tensordot(basis_full, coeffs.astype(np.float32), axes=([2], [0]))
        mean_surf = float(np.mean(cup_surface[::stride, ::stride].ravel()[m_sub]))
        out[iz] = sl - (cup_surface - np.float32(mean_surf))

    out_arr = out[0] if is_2d else out
    if isinstance(vol_zyx, Volume):
        prov = stamp_provenance(vol_zyx.provenance)
        steps = list(prov.get("processing_steps", []))
        steps.append({"step": "detrend_radial_cupping"})
        prov["processing_steps"] = steps
        return Volume(
            out_arr,
            vol_zyx.axes,
            space=vol_zyx.space,
            centering=vol_zyx.centering,
            provenance=prov,
        )
    return out_arr
