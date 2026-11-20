"""Tests for HDF5, TIFF, .npy, and .raw volume round-trips and region reads."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from basecoat import Volume, __version__, read_volume, write_volume


def test_hdf5_round_trip_and_multidataset_disambiguation(tmp_path: Path) -> None:
    arr = np.linspace(-1.0, 2.0, 10 * 12 * 14, dtype=np.float32).reshape(10, 12, 14)
    vol = Volume.from_array(
        arr, voxel_size_um=(2.0, 1.0, 1.0), origin_um=(4.0, 2.0, 0.0), space="h5_space"
    )
    h5_path = write_volume(vol, tmp_path / "sample.h5", dataset_path="recon/tomo")

    rt = read_volume(h5_path)
    assert rt.shape == (10, 12, 14)
    assert rt.space == "h5_space"
    assert np.allclose(rt.spacing_in("um"), (2.0, 1.0, 1.0))
    assert np.allclose(rt.data, arr)
    assert rt.provenance["basecoat_version"] == __version__

    # Multi-dataset HDF5 requires explicit dataset_path
    multi_h5 = tmp_path / "multi.h5"
    with h5py.File(multi_h5, "w") as h5f:
        h5f.create_dataset("scan_a", data=arr)
        h5f.create_dataset("scan_b", data=arr * 2.0)

    with pytest.raises(ValueError, match="dataset_path"):
        read_volume(multi_h5)

    explicit = read_volume(multi_h5, dataset_path="scan_b", voxel_size_um=0.5)
    assert np.allclose(explicit.data, arr * 2.0)
    assert np.allclose(explicit.spacing_in("um"), (0.5, 0.5, 0.5))


def test_tiff_npy_and_raw_round_trips_and_region_slicing(tmp_path: Path) -> None:
    arr = np.arange(8 * 10 * 12, dtype=np.uint16).reshape(8, 10, 12)
    vol = Volume.from_array(arr, voxel_size_um=0.325, origin_um=(1.0, 2.0, 3.0), space="i13_10x")

    # TIFF round-trip & slice-region read
    tif_path = write_volume(vol, tmp_path / "stack.tif")
    tif_sub = read_volume(tif_path, region=(slice(1, 5), slice(2, 8), slice(3, 9)))
    assert tif_sub.shape == (4, 6, 6)
    assert np.array_equal(tif_sub.data, arr[1:5, 2:8, 3:9])
    assert tif_sub.space == "i13_10x"
    assert np.isclose(tif_sub.axes[0].offset, 1.0 + 1 * 0.325)

    # .npy memmap region read
    npy_path = write_volume(vol, tmp_path / "cube.npy")
    npy_sub = read_volume(
        npy_path, region=(slice(0, 3), slice(0, 4), slice(0, 5)), voxel_size_um=0.325
    )
    assert np.array_equal(npy_sub.data, arr[0:3, 0:4, 0:5])

    # .raw binary requires explicit raw_shape and raw_dtype
    raw_path = write_volume(vol, tmp_path / "cube.raw")
    with pytest.raises(ValueError, match="raw_shape"):
        read_volume(raw_path)

    raw_rt = read_volume(
        raw_path,
        raw_shape=(8, 10, 12),
        raw_dtype="uint16_le",
        region=(slice(2, 6), slice(1, 5), slice(4, 10)),
        voxel_size_um=0.325,
    )
    assert np.array_equal(raw_rt.data, arr[2:6, 1:5, 4:10])
