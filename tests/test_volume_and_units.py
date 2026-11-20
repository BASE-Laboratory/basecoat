"""Tests for AxisSpec, single-array Volume, co-registered VolumeSet, and unit safety."""

from __future__ import annotations

import numpy as np
import pytest

from basecoat import (
    SpaceMismatchError,
    UnitConversionError,
    Volume,
    VolumeProtocol,
    VolumeSet,
    coarsen_axes,
)


def test_volume_protocol_and_unit_conversion() -> None:
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    vol = Volume.from_array(
        arr, voxel_size_um=(1.5, 0.8, 0.8), origin_um=(10.0, 20.0, 30.0), space="scan_a"
    )

    assert isinstance(vol, VolumeProtocol)
    assert vol.is_calibrated
    assert vol.shape == (2, 3, 4)
    assert vol.axis_names == ("z", "y", "x")
    assert np.allclose(vol.spacing_in("um"), (1.5, 0.8, 0.8))
    assert np.allclose(vol.spacing_in("m"), (1.5e-6, 0.8e-6, 0.8e-6))
    assert np.allclose(vol.spacing_in("nm"), (1500.0, 800.0, 800.0))
    assert np.allclose(vol.origin_in("m"), (10.0e-6, 20.0e-6, 30.0e-6))

    # NumPy 1.26 and 2.x __array__ compatibility
    as_arr = np.asarray(vol)
    assert np.shares_memory(as_arr, arr)
    copied = np.array(vol, dtype=np.float64, copy=True)
    assert copied.dtype == np.float64
    assert not np.shares_memory(copied, arr)


def test_uncalibrated_volume_refuses_physical_conversion_with_actionable_error() -> None:
    vol = Volume.uncalibrated(np.zeros((4, 4, 4), dtype=np.uint8))
    assert not vol.is_calibrated
    with pytest.raises(UnitConversionError, match="voxel_size_um"):
        vol.spacing_in("m")

    calibrated = vol.with_voxel_size_um(0.5)
    assert calibrated.is_calibrated
    assert np.allclose(calibrated.spacing_in("m"), (5e-7, 5e-7, 5e-7))


def test_cell_centre_slicing_and_coarsen_axes_outer_walls() -> None:
    arr = np.zeros((16, 20, 24), dtype=np.float32)
    vol = Volume.from_array(arr, voxel_size_um=(2.0, 1.0, 0.5), origin_um=(100.0, -50.0, 10.0))

    sub = vol[3:11, 5:15, 4:20:2]
    assert sub.shape == (8, 10, 8)
    # Index 0 of sub corresponds to index (3, 5, 4) of vol:
    assert np.isclose(sub.axes[0].offset, 100.0 + 3 * 2.0)
    assert np.isclose(sub.axes[1].offset, -50.0 + 5 * 1.0)
    assert np.isclose(sub.axes[2].offset, 10.0 + 4 * 0.5)
    assert np.isclose(sub.axes[2].scale, 0.5 * 2)

    # Outer voxel left wall at (offset - 0.5 * scale) is preserved under k-factor coarsening
    coarse = coarsen_axes(vol.axes, factors=4)
    for fine_ax, coarse_ax in zip(vol.axes, coarse, strict=True):
        fine_left_wall = fine_ax.offset - 0.5 * fine_ax.scale
        coarse_left_wall = coarse_ax.offset - 0.5 * coarse_ax.scale
        assert np.isclose(fine_left_wall, coarse_left_wall, atol=1e-12)
        assert np.isclose(coarse_ax.scale, 4.0 * fine_ax.scale)


def test_zero_copy_fortran_view() -> None:
    arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    vol = Volume.from_array(arr, voxel_size_um=1.0)
    f_view = vol.fortran_view()
    assert f_view.shape == (5, 4, 3)
    assert f_view.flags.f_contiguous
    assert np.shares_memory(f_view, arr)
    assert f_view[4, 2, 1] == arr[1, 2, 4]


def test_volumeset_enforces_coregistration_and_slices_in_lockstep() -> None:
    img = Volume.from_array(np.ones((8, 10, 12), dtype=np.float32), voxel_size_um=0.8, space="s1")
    lbl = Volume.from_array(
        np.zeros((8, 10, 12), dtype=np.int32),
        voxel_size_um=0.8,
        space="s1",
        label_names={0: "pore", 1: "solid"},
    )
    vset = VolumeSet(img, labels={"phase": lbl})
    cropped = vset[2:6, 1:9, 3:11]
    assert cropped.image.shape == (4, 8, 8)
    assert cropped.labels["phase"].shape == (4, 8, 8)
    assert cropped.labels["phase"].axes == cropped.image.axes

    # Reject float label volume
    bad_float_lbl = Volume.from_array(
        np.zeros((8, 10, 12), dtype=np.float32), voxel_size_um=0.8, space="s1"
    )
    with pytest.raises(ValueError, match="integer dtype"):
        VolumeSet(img, labels={"phase": bad_float_lbl})

    # Reject mismatched voxel size
    bad_scale_lbl = Volume.from_array(
        np.zeros((8, 10, 12), dtype=np.int32), voxel_size_um=0.9, space="s1"
    )
    with pytest.raises(SpaceMismatchError):
        VolumeSet(img, labels={"phase": bad_scale_lbl})

    # Reject mismatched space
    bad_space_lbl = Volume.from_array(
        np.zeros((8, 10, 12), dtype=np.int32), voxel_size_um=0.8, space="s2"
    )
    with pytest.raises(SpaceMismatchError):
        VolumeSet(img, labels={"phase": bad_space_lbl})
