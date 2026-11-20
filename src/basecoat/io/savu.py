"""Out-of-core Diamond Light Source Savu / TomoPy / Astra HDF5 reader.

Handles Savu's ``(X, Z_vertical, Y)`` on-disk axis permutation transparently so callers
always slice and receive volumes in canonical ``('z', 'y', 'x')`` order without
materialising >50 GB reconstructions in host memory.

Adapted from ``MultimodalBenchmark/src/multimodal_benchmark/io.py`` and
``OpenLSR-X/src/openlsrx/data/zn13x.py`` (James Le Houx, Calum Green, Sharif Ahmed).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..volume import AxisSpec, LazyVolume, Volume

SAVU_DATASET_CANDIDATES: tuple[str, ...] = (
    "4-TomopyRecon-tomo/data",
    "4-AstraReconGpu-tomo/data",
    "entry/final_result_tomo/data",
    "data",
    "volume",
)

_CANONICAL_SAVU_SHAPES: set[tuple[int, int, int]] = {
    (2510, 2110, 2510),
    (2560, 2160, 2560),
}


def find_savu_dataset_path(h5_file: h5py.File) -> str:
    """Locate the primary 3-D tomography dataset inside a Savu or cropped HDF5 file."""
    for candidate in SAVU_DATASET_CANDIDATES:
        if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
            return candidate

    found: list[str] = []

    def _visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found.append(name)

    h5_file.visititems(_visitor)
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(f"No 3-D tomography dataset found in HDF5 file '{h5_file.filename}'.")
    raise ValueError(
        f"Multiple 3-D datasets {found} found in '{h5_file.filename}'; pass dataset_path='...' explicitly."
    )


def is_savu_xzy_layout(dataset_path: str, raw_shape: Sequence[int]) -> bool:
    """Return True if an HDF5 dataset is stored in Savu's ``(X, Z_vertical, Y)`` layout."""
    if len(raw_shape) != 3:
        return False
    shp = (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2]))
    if any(tag in dataset_path for tag in ("TomopyRecon", "AstraRecon")):
        return shp[0] == shp[2] and shp[1] != shp[0]
    return shp in _CANONICAL_SAVU_SHAPES


def open_savu_h5(
    path: str | Path,
    *,
    dataset_path: str | None = None,
    savu_xzy: bool | None = None,
    voxel_size_um: float | Sequence[float] | None = None,
    origin_um: Sequence[float] | None = None,
    space: str | None = None,
) -> LazyVolume:
    """Open a Savu or cropped tomography HDF5 file lazily in canonical ``('z', 'y', 'x')`` order."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Savu HDF5 path does not exist: '{p}'.")

    h5f = h5py.File(p, mode="r")
    try:
        ds_name = dataset_path if dataset_path is not None else find_savu_dataset_path(h5f)
        if ds_name not in h5f or not isinstance(h5f[ds_name], h5py.Dataset):
            raise KeyError(f"HDF5 dataset '{ds_name}' not found in '{p}'.")
        ds = h5f[ds_name]
        if ds.ndim != 3:
            raise ValueError(
                f"open_savu_h5 expects a 3-D tomography dataset, got shape={ds.shape} at '{ds_name}'."
            )
        raw_shape = (int(ds.shape[0]), int(ds.shape[1]), int(ds.shape[2]))
        use_xzy = is_savu_xzy_layout(ds_name, raw_shape) if savu_xzy is None else bool(savu_xzy)
        zyx_shape = (raw_shape[1], raw_shape[2], raw_shape[0]) if use_xzy else raw_shape

        dummy = np.empty((1, 1, 1), dtype=np.uint8)
        axes = Volume.from_array(
            dummy,
            voxel_size_um=voxel_size_um,
            origin_um=origin_um,
            axes_names=("z", "y", "x"),
            space=space,
        ).axes
        dtype = np.dtype(ds.dtype)
        prov: dict[str, Any] = {
            "source_format": "savu_hdf5",
            "dataset_path": ds_name,
            "savu_xzy_transposed": use_xzy,
        }
    except Exception:
        h5f.close()
        raise

    def _read_zyx_slices(slices: tuple[slice, ...]) -> np.ndarray[Any, Any]:
        sz, sy, sx = slices
        if use_xzy:
            raw = np.asarray(ds[(sx, sz, sy)])
            return np.ascontiguousarray(np.transpose(raw, (1, 2, 0)))
        return np.ascontiguousarray(np.asarray(ds[(sz, sy, sx)]))

    return LazyVolume(
        shape=zyx_shape,
        dtype=dtype,
        axes=axes,
        reader=_read_zyx_slices,
        closer=h5f.close,
        space=space,
        provenance=prov,
    )


def read_savu_subvolume(
    path: str | Path,
    *,
    zyx_slices: tuple[slice, slice, slice] | None = None,
    stride: int = 1,
    flip_x: bool = False,
    dataset_path: str | None = None,
    savu_xzy: bool | None = None,
    voxel_size_um: float | Sequence[float] | None = None,
    origin_um: Sequence[float] | None = None,
    space: str | None = None,
    dtype: np.dtype[Any] | type[Any] | None = None,
) -> Volume:
    """Read a 3-D sub-volume out-of-core in canonical ``('z', 'y', 'x')`` order."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    with open_savu_h5(
        path,
        dataset_path=dataset_path,
        savu_xzy=savu_xzy,
        voxel_size_um=voxel_size_um,
        origin_um=origin_um,
        space=space,
    ) as lv:
        if zyx_slices is None:
            eff_slices = (
                slice(None, None, stride),
                slice(None, None, stride),
                slice(None, None, stride),
            )
        else:
            sz, sy, sx = zyx_slices
            eff_slices = (
                slice(sz.start, sz.stop, sz.step if sz.step is not None else stride),
                slice(sy.start, sy.stop, sy.step if sy.step is not None else stride),
                slice(sx.start, sx.stop, sx.step if sx.step is not None else stride),
            )
        vol = lv.read_region(eff_slices)

    arr = vol.data
    axes = list(vol.axes)
    prov = dict(vol.provenance)

    if flip_x:
        arr = np.flip(arr, axis=2).copy()
        ax_x = axes[2]
        nx = int(arr.shape[2])
        # Reversing axis x places the former last cell centre (offset + (nx - 1)*scale) at index 0
        axes[2] = AxisSpec(
            name=ax_x.name,
            kind=ax_x.kind,
            unit=ax_x.unit,
            scale=ax_x.scale,
            offset=ax_x.offset + max(0, nx - 1) * ax_x.scale,
        )
        prov["flip_x_applied"] = True

    if dtype is not None:
        target_dt = np.dtype(dtype)
        if arr.dtype != target_dt:
            arr = arr.astype(target_dt, copy=False)

    return Volume(
        arr,
        axes=axes,
        space=vol.space,
        centering=vol.centering,
        label_names=vol.label_names,
        provenance=prov,
    )
