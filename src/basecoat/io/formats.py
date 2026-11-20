"""Generic HDF5, TIFF, raw binary, and NumPy (.npy) volume readers, writers, and lazy handles.

Portions of the 1-bit packed TIFF decoder and raw dtype table are adapted from
``BASE-Laboratory/OpenImpala`` (BSD-3-Clause, James Le Houx).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile

from .._provenance import stamp_provenance
from ..volume import AxisSpec, Centering, LazyVolume, Volume
from .ome_zarr import open_ome_zarr, read_ome_zarr, write_ome_zarr

_RAW_DTYPE_ALIASES: dict[str, str] = {
    "uint8": "<u1",
    "int8": "<i1",
    "uint16": "<u2",
    "uint16_le": "<u2",
    "uint16_be": ">u2",
    "int16": "<i2",
    "int16_le": "<i2",
    "int16_be": ">i2",
    "uint32": "<u4",
    "uint32_le": "<u4",
    "uint32_be": ">u4",
    "int32": "<i4",
    "int32_le": "<i4",
    "int32_be": ">i4",
    "float32": "<f4",
    "float32_le": "<f4",
    "float32_be": ">f4",
    "float64": "<f8",
    "float64_le": "<f8",
    "float64_be": ">f8",
}


def _is_zarr_path(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".zarr")
        or name.endswith(".ome.zarr")
        or (path.is_dir() and ((path / ".zgroup").exists() or (path / ".zarray").exists()))
    )


def _serialize_volume_attrs(vol: Volume) -> dict[str, str]:
    stamped = stamp_provenance(vol.provenance)
    axes_list = [
        {
            "name": ax.name,
            "kind": ax.kind,
            "unit": ax.unit,
            "scale": float(ax.scale),
            "offset": float(ax.offset),
        }
        for ax in vol.axes
    ]
    meta: dict[str, Any] = {
        "schema": "basecoat.volume.v1",
        "axes": axes_list,
        "space": vol.space,
        "centering": vol.centering,
        "label_names": {str(k): v for k, v in vol.label_names.items()} if vol.label_names else None,
        "provenance": stamped,
    }
    return {"basecoat_metadata_json": json.dumps(meta)}


def _deserialize_volume_attrs(
    raw_json: str | bytes | None,
    ndim: int,
    *,
    voxel_size_um: float | Sequence[float] | None = None,
    origin_um: Sequence[float] | None = None,
    space: str | None = None,
) -> tuple[
    tuple[AxisSpec, ...],
    str | None,
    Centering,
    dict[int, str] | None,
    dict[str, Any],
]:
    parsed_axes: tuple[AxisSpec, ...] | None = None
    parsed_space: str | None = None
    parsed_centering: Centering = "cell"
    parsed_labels: dict[int, str] | None = None
    parsed_prov: dict[str, Any] = {}

    if raw_json is not None:
        text = raw_json.decode("utf-8") if isinstance(raw_json, bytes) else raw_json
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("schema") == "basecoat.volume.v1":
                raw_axes = payload.get("axes", [])
                if isinstance(raw_axes, list) and len(raw_axes) == ndim:
                    parsed_axes = tuple(
                        AxisSpec(
                            name=str(a["name"]),
                            kind=a["kind"],
                            unit=str(a["unit"]),
                            scale=float(a["scale"]),
                            offset=float(a["offset"]),
                        )
                        for a in raw_axes
                    )
                sp = payload.get("space")
                parsed_space = str(sp) if isinstance(sp, str) else None
                cnt = payload.get("centering", "cell")
                if cnt in ("cell", "node", "face_x", "face_y", "face_z"):
                    parsed_centering = cnt
                ln = payload.get("label_names")
                if isinstance(ln, dict):
                    parsed_labels = {int(k): str(v) for k, v in ln.items()}
                pr = payload.get("provenance")
                if isinstance(pr, dict):
                    parsed_prov = dict(pr)
        except (ValueError, KeyError, TypeError):
            pass

    if voxel_size_um is not None or parsed_axes is None:
        dummy = np.empty((1,) * ndim, dtype=np.uint8)
        helper = Volume.from_array(
            dummy,
            voxel_size_um=voxel_size_um,
            origin_um=origin_um,
            space=space if space is not None else parsed_space,
        )
        parsed_axes = helper.axes

    final_space = space if space is not None else parsed_space
    return parsed_axes, final_space, parsed_centering, parsed_labels, parsed_prov


def _resolve_h5_dataset_path(h5f: h5py.File, dataset_path: str | None) -> str:
    if dataset_path is not None:
        if dataset_path not in h5f or not isinstance(h5f[dataset_path], h5py.Dataset):
            raise KeyError(f"HDF5 dataset '{dataset_path}' not found in '{h5f.filename}'.")
        return dataset_path

    if "volume" in h5f and isinstance(h5f["volume"], h5py.Dataset):
        return "volume"

    found: list[str] = []

    def _collect(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            found.append(name)

    h5f.visititems(_collect)
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(f"No >=2-D HDF5 datasets found in '{h5f.filename}'.")
    raise ValueError(
        f"HDF5 file '{h5f.filename}' contains multiple datasets {found}; pass dataset_path='...' explicitly."
    )


def open_volume(
    path: str | Path,
    *,
    dataset_path: str | None = None,
    voxel_size_um: float | Sequence[float] | None = None,
    origin_um: Sequence[float] | None = None,
    space: str | None = None,
    raw_shape: Sequence[int] | None = None,
    raw_dtype: str | np.dtype[Any] | None = None,
    level: int = 0,
) -> LazyVolume:
    """Open an OME-Zarr, HDF5, `.npy`, TIFF, or `.raw` volume lazily for region slicing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Volume path does not exist: '{p}'.")

    if _is_zarr_path(p):
        lv = open_ome_zarr(p, level=level)
        if voxel_size_um is None and space is None:
            return lv
        dummy = np.empty((1,) * lv.ndim, dtype=np.uint8)
        calibrated = Volume.from_array(
            dummy,
            voxel_size_um=voxel_size_um,
            origin_um=origin_um,
            axes_names=lv.axis_names,
            space=space if space is not None else lv.space,
        )
        return LazyVolume(
            shape=lv.shape,
            dtype=lv.dtype,
            axes=calibrated.axes if voxel_size_um is not None else lv.axes,
            reader=lambda slc: lv.read_region(slc).data,
            closer=lv.close,
            space=space if space is not None else lv.space,
        )

    suffix = p.suffix.lower()

    if suffix in (".h5", ".hdf5", ".nxs"):
        h5f = h5py.File(p, mode="r")
        try:
            ds_key = _resolve_h5_dataset_path(h5f, dataset_path)
            ds = h5f[ds_key]
            raw_meta = ds.attrs.get("basecoat_metadata_json") or h5f.attrs.get(
                "basecoat_metadata_json"
            )
            axes, sp, cnt, lbls, prov = _deserialize_volume_attrs(
                raw_meta,
                int(ds.ndim),
                voxel_size_um=voxel_size_um,
                origin_um=origin_um,
                space=space,
            )
            shape = tuple(int(s) for s in ds.shape)
            dtype = np.dtype(ds.dtype)
        except Exception:
            h5f.close()
            raise

        def _read_h5(slices: tuple[slice, ...]) -> np.ndarray[Any, Any]:
            return np.asarray(ds[slices])

        return LazyVolume(
            shape=shape,
            dtype=dtype,
            axes=axes,
            reader=_read_h5,
            closer=h5f.close,
            space=sp,
            centering=cnt,
            label_names=lbls,
            provenance=prov,
        )

    if suffix == ".npy":
        mmap_arr = np.load(p, mmap_mode="r")
        axes, sp, cnt, lbls, prov = _deserialize_volume_attrs(
            None,
            int(mmap_arr.ndim),
            voxel_size_um=voxel_size_um,
            origin_um=origin_um,
            space=space,
        )

        def _read_npy(slices: tuple[slice, ...]) -> np.ndarray[Any, Any]:
            return np.array(mmap_arr[slices], copy=True)

        def _close_npy() -> None:
            mm = getattr(mmap_arr, "_mmap", None)
            if mm is not None:
                mm.close()

        return LazyVolume(
            shape=tuple(int(s) for s in mmap_arr.shape),
            dtype=np.dtype(mmap_arr.dtype),
            axes=axes,
            reader=_read_npy,
            closer=_close_npy,
            space=sp,
            centering=cnt,
            label_names=lbls,
            provenance=prov,
        )

    if suffix in (".tif", ".tiff"):
        tf = tifffile.TiffFile(p)
        try:
            series = tf.series[0]
            shape = tuple(int(s) for s in series.shape)
            dtype = np.dtype(series.dtype)
            desc: str | None = None
            if len(tf.pages) > 0:
                raw_desc = getattr(tf.pages[0], "description", None)
                if isinstance(raw_desc, str) and raw_desc:
                    desc = raw_desc
            axes, sp, cnt, lbls, prov = _deserialize_volume_attrs(
                desc,
                len(shape),
                voxel_size_um=voxel_size_um,
                origin_um=origin_um,
                space=space,
            )
        except Exception:
            tf.close()
            raise

        def _read_tiff(slices: tuple[slice, ...]) -> np.ndarray[Any, Any]:
            if len(shape) == 3 and len(tf.pages) == shape[0]:
                z_slc, y_slc, x_slc = slices
                z_indices = range(*z_slc.indices(shape[0]))
                planes = [tf.pages[idx].asarray()[y_slc, x_slc] for idx in z_indices]
                return np.stack(planes, axis=0)
            full = series.asarray()
            return np.asarray(full[slices])

        return LazyVolume(
            shape=shape,
            dtype=dtype,
            axes=axes,
            reader=_read_tiff,
            closer=tf.close,
            space=sp,
            centering=cnt,
            label_names=lbls,
            provenance=prov,
        )

    if suffix == ".raw":
        if raw_shape is None or raw_dtype is None:
            raise ValueError(
                f"Opening raw binary volume '{p}' requires both raw_shape=(nz, ny, nx) and raw_dtype='...'."
            )
        dt_spec = (
            _RAW_DTYPE_ALIASES.get(raw_dtype, raw_dtype)
            if isinstance(raw_dtype, str)
            else raw_dtype
        )
        dt = np.dtype(dt_spec)
        shp = tuple(int(s) for s in raw_shape)
        mm = np.memmap(p, dtype=dt, mode="r", shape=shp)
        axes, sp, cnt, lbls, prov = _deserialize_volume_attrs(
            None,
            len(shp),
            voxel_size_um=voxel_size_um,
            origin_um=origin_um,
            space=space,
        )

        def _read_raw(slices: tuple[slice, ...]) -> np.ndarray[Any, Any]:
            return np.array(mm[slices], copy=True)

        def _close_raw() -> None:
            inner = getattr(mm, "_mmap", None)
            if inner is not None:
                inner.close()

        return LazyVolume(
            shape=shp,
            dtype=dt,
            axes=axes,
            reader=_read_raw,
            closer=_close_raw,
            space=sp,
            centering=cnt,
            label_names=lbls,
            provenance=prov,
        )

    raise ValueError(
        f"Unsupported volume format for '{p}'. Supported extensions: .ome.zarr, .h5/.hdf5/.nxs, .tif/.tiff, .npy, .raw."
    )


def read_volume(
    path: str | Path,
    *,
    region: slice | tuple[slice | Any, ...] | None = None,
    dataset_path: str | None = None,
    voxel_size_um: float | Sequence[float] | None = None,
    origin_um: Sequence[float] | None = None,
    space: str | None = None,
    raw_shape: Sequence[int] | None = None,
    raw_dtype: str | np.dtype[Any] | None = None,
    level: int = 0,
    max_bytes: int = 2 * 1024**3,
) -> Volume:
    """Read a full volume or bounded *region* into a :class:`Volume`."""
    p = Path(path)
    if _is_zarr_path(p) and region is None and voxel_size_um is None and space is None:
        return read_ome_zarr(p, level=level, as_set=False)

    with open_volume(
        p,
        dataset_path=dataset_path,
        voxel_size_um=voxel_size_um,
        origin_um=origin_um,
        space=space,
        raw_shape=raw_shape,
        raw_dtype=raw_dtype,
        level=level,
    ) as lv:
        if region is not None:
            return lv.read_region(region)
        return lv.to_volume(max_bytes=max_bytes)


def write_volume(
    vol: Volume,
    path: str | Path,
    *,
    dataset_path: str = "volume",
    chunks: Sequence[int] | bool = True,
    compression: str | None = "gzip",
    overwrite: bool = False,
) -> Path:
    """Write a :class:`Volume` to `.ome.zarr`, `.h5`/`.hdf5`, `.tif`/`.tiff`, `.npy`, or `.raw`."""
    out = Path(path)
    if _is_zarr_path(out) or out.name.lower().endswith((".zarr", ".ome.zarr")):
        chunk_seq = tuple(int(c) for c in chunks) if isinstance(chunks, Sequence) else None
        return write_ome_zarr(vol, out, chunks=chunk_seq, overwrite=overwrite)

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{out}' already exists; pass overwrite=True to replace it."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    attrs = _serialize_volume_attrs(vol)

    if suffix in (".h5", ".hdf5", ".nxs"):
        with h5py.File(out, mode="w") as h5f:
            h5_chunks: tuple[int, ...] | bool = (
                tuple(int(c) for c in chunks) if isinstance(chunks, Sequence) else bool(chunks)
            )
            ds = h5f.create_dataset(
                dataset_path,
                data=vol.data,
                chunks=h5_chunks,
                compression=compression,
            )
            for k, v in attrs.items():
                ds.attrs[k] = v
                h5f.attrs[k] = v
        return out

    if suffix in (".tif", ".tiff"):
        tifffile.imwrite(
            out,
            vol.data,
            description=attrs["basecoat_metadata_json"],
            compression="zlib" if compression == "gzip" else compression,
        )
        return out

    if suffix == ".npy":
        np.save(out, vol.data)
        return out

    if suffix == ".raw":
        np.ascontiguousarray(vol.data).tofile(out)
        return out

    raise ValueError(
        f"Unsupported output extension '{suffix}' for '{out}'. Supported: .ome.zarr, .h5/.hdf5, .tif/.tiff, .npy, .raw."
    )
