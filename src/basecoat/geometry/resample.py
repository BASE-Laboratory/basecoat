"""Cell-centred 3-D volume resampling under the index-0 cell-centre convention."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import affine_transform

from ..exceptions import SpaceMismatchError
from ..volume import AxisSpec, Volume, convert_length
from .transforms import AffineTransform3D, SimilarityTransform3D


def _extract_3d_spatial_axes(
    axes: Sequence[AxisSpec],
    target_unit: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if len(axes) != 3 or tuple(ax.name for ax in axes) != ("z", "y", "x"):
        raise ValueError(
            f"resample_volume requires 3-D ('z', 'y', 'x') spatial axes, got {tuple(ax.name for ax in axes)}."
        )
    if target_unit == "voxel":
        if any(ax.unit != "voxel" for ax in axes):
            raise ValueError(
                "Transform has unit='voxel', so both moving and target volumes must have unit='voxel'."
            )
        scales = np.array([ax.scale for ax in axes], dtype=np.float64)
        offsets = np.array([ax.offset for ax in axes], dtype=np.float64)
        return scales, offsets

    scales = np.array(
        [convert_length(ax.scale, ax.unit, target_unit, axis_name=ax.name) for ax in axes],
        dtype=np.float64,
    )
    offsets = np.array(
        [convert_length(ax.offset, ax.unit, target_unit, axis_name=ax.name) for ax in axes],
        dtype=np.float64,
    )
    return scales, offsets


def resample_volume(
    moving: Volume,
    target_axes: Sequence[AxisSpec] | Volume,
    *,
    target_shape: tuple[int, int, int] | None = None,
    transform: AffineTransform3D | SimilarityTransform3D | None = None,
    order: int | None = None,
    cval: float | None = None,
    target_space: str | None = None,
) -> Volume:
    """Resample *moving* onto a target 3-D ``('z', 'y', 'x')`` grid under index-0 cell-centre geometry.

    Parameters
    ----------
    moving : Volume
        Input 3-D ``('z', 'y', 'x')`` volume to sample from.
    target_axes : Sequence[AxisSpec] | Volume
        Either a reference :class:`Volume` (providing both ``shape``, ``axes``, and ``space``)
        or a 3-element ``AxisSpec`` sequence (requiring *target_shape*).
    target_shape : tuple[int, int, int] | None
        Output ``(nz, ny, nx)`` shape when *target_axes* is a sequence of :class:`AxisSpec`.
    transform : AffineTransform3D | SimilarityTransform3D | None
        Pullback transform mapping ``target (fixed) -> moving`` coordinates. When ``None``,
        uses the identity mapping in physical space (`"um"` if calibrated, `"voxel"` if both are uncalibrated).
    order : int | None
        Spline interpolation order. Defaults to ``0`` (nearest-neighbour) for integer label
        volumes and ``1`` (trilinear) for floating-point volumes.
    cval : float | None
        Fill value outside the moving domain. Defaults to ``0`` for integer volumes and the
        5th percentile of *moving* (matching ``MultimodalBenchmark``) when ``None`` is not overridden;
        pass ``cval=0.0`` explicitly when zero padding is desired.
    target_space : str | None
        Space tag for the returned volume (defaults to reference ``Volume.space`` or ``transform.from_space``).
    """
    if isinstance(target_axes, Volume):
        if target_axes.ndim != 3:
            raise ValueError(f"Reference target Volume must be 3-D, got shape={target_axes.shape}.")
        out_axes = target_axes.axes
        shp = (int(target_axes.shape[0]), int(target_axes.shape[1]), int(target_axes.shape[2]))
        ref_space = target_axes.space
    else:
        if target_shape is None or len(target_shape) != 3:
            raise ValueError(
                "target_shape=(nz, ny, nx) is required when target_axes is a sequence of AxisSpec."
            )
        out_axes = tuple(target_axes)
        shp = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
        ref_space = None

    aff: AffineTransform3D | None = (
        transform.to_affine() if isinstance(transform, SimilarityTransform3D) else transform
    )

    if aff is not None:
        if aff.axis_order != ("z", "y", "x"):
            raise SpaceMismatchError(
                f"resample_volume requires transform axis_order=('z', 'y', 'x'), got {aff.axis_order}."
            )
        if moving.space is not None and moving.space != aff.to_space:
            raise SpaceMismatchError(
                f"Transform to_space={aff.to_space!r} does not match moving.space={moving.space!r}."
            )
        if ref_space is not None and ref_space != aff.from_space:
            raise SpaceMismatchError(
                f"Transform from_space={aff.from_space!r} does not match target.space={ref_space!r}."
            )
        work_unit = aff.unit
        a_phys = aff.linear_matrix
        b_phys = aff.translation_vector
        out_space: str | None = target_space or ref_space or aff.from_space
    else:
        if moving.space is not None and ref_space is not None and moving.space != ref_space:
            raise SpaceMismatchError(
                f"Cannot identity-resample between distinct spaces ({ref_space!r} and {moving.space!r}) "
                f"without an explicit transform."
            )
        work_unit = (
            "voxel"
            if (not moving.is_calibrated and all(ax.unit == "voxel" for ax in out_axes))
            else "um"
        )
        a_phys = np.eye(3, dtype=np.float64)
        b_phys = np.zeros(3, dtype=np.float64)
        out_space = target_space or ref_space or moving.space

    s_mov, o_mov = _extract_3d_spatial_axes(moving.axes, work_unit)
    s_tgt, o_tgt = _extract_3d_spatial_axes(out_axes, work_unit)

    inv_s_mov = np.diag(1.0 / s_mov)
    diag_s_tgt = np.diag(s_tgt)

    # Index-0 cell-centre algebra:
    # x_tgt(i_out) = o_tgt + S_tgt @ i_out
    # x_mov        = A_phys @ x_tgt + b_phys
    # i_in         = S_mov^{-1} @ (x_mov - o_mov) = A_idx @ i_out + b_idx
    a_idx = inv_s_mov @ a_phys @ diag_s_tgt
    b_idx = inv_s_mov @ (a_phys @ o_tgt + b_phys - o_mov)

    is_int = np.issubdtype(moving.dtype, np.integer)
    interp_order = (0 if is_int else 1) if order is None else int(order)
    if is_int and interp_order != 0:
        raise ValueError(
            f"Resampling integer label volume with order={interp_order} would invent intermediate class IDs; "
            f"use order=0 or cast moving data to float explicitly."
        )

    fill_val = (
        (0.0 if is_int else float(np.percentile(moving.data, 5.0))) if cval is None else float(cval)
    )
    src_arr = (
        moving.data
        if is_int
        else np.asarray(moving.data, dtype=np.float32 if moving.dtype == np.float32 else np.float64)
    )

    resampled = affine_transform(
        src_arr,
        matrix=a_idx,
        offset=b_idx,
        output_shape=shp,
        order=interp_order,
        mode="constant",
        cval=int(round(fill_val)) if is_int else fill_val,
        prefilter=(interp_order > 1),
    )

    out_dtype = (
        moving.dtype if is_int else (np.float64 if moving.dtype == np.float64 else np.float32)
    )
    prov = dict(moving.provenance)
    prov["resampled"] = {
        "order": interp_order,
        "from_space": out_space,
        "to_space": moving.space,
    }
    return Volume(
        np.asarray(resampled, dtype=out_dtype),
        out_axes,
        space=out_space,
        centering=moving.centering,
        label_names=moving.label_names if is_int else None,
        provenance=prov,
    )
