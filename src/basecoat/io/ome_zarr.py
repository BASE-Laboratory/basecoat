"""OME-Zarr v0.4 (Zarr v2) reader, writer, and lazy handle.

Engineering rule: This is the **only** module in ``basecoat`` permitted to import
``zarr`` or ``numcodecs`` (enforced by ``tests/test_zarr_isolation.py``), isolating
any future migration to Zarr v3 / OME-Zarr v0.5 to this file.

Portions of the OME-Zarr v0.4 multiscale writer build on packaging utilities
authored by James Le Houx and Calum Green in ``MultimodalBenchmark``.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast, overload

import numcodecs
import numpy as np
import zarr

from .._provenance import stamp_provenance
from ..volume import (
    AxisKind,
    AxisSpec,
    Centering,
    LazyVolume,
    Volume,
    VolumeSet,
    coarsen_axes,
)

_CANONICAL_TO_NGFF_SPACE_UNIT: dict[str, str] = {
    "nm": "nanometer",
    "um": "micrometer",
    "mm": "millimeter",
    "cm": "centimeter",
    "m": "meter",
    "angstrom": "angstrom",
}

_NGFF_TO_CANONICAL_UNIT: dict[str, str] = {
    "nanometer": "nm",
    "micrometer": "um",
    "millimeter": "mm",
    "centimeter": "cm",
    "meter": "m",
    "angstrom": "angstrom",
    "second": "s",
    "millisecond": "ms",
}

_CANONICAL_TO_NGFF_TIME_UNIT: dict[str, str] = {
    "s": "second",
    "ms": "millisecond",
}

_DEFAULT_LABEL_PALETTE_RGBA: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 0),
    (230, 159, 0, 255),
    (86, 180, 233, 255),
    (0, 158, 115, 255),
    (240, 228, 66, 255),
    (0, 114, 178, 255),
    (213, 94, 0, 255),
    (204, 121, 167, 255),
)


def _pad_for_factor2(
    arr: np.ndarray[Any, Any],
    spatial_flags: Sequence[bool],
) -> tuple[np.ndarray[Any, Any], tuple[int, ...]]:
    pad_widths: list[tuple[int, int]] = []
    factors: list[int] = []
    needs_pad = False
    for length, is_space in zip(arr.shape, spatial_flags, strict=True):
        if is_space and length > 1:
            rem = length % 2
            pad_widths.append((0, rem))
            factors.append(2)
            if rem != 0:
                needs_pad = True
        else:
            pad_widths.append((0, 0))
            factors.append(1)
    padded = np.pad(arr, pad_widths, mode="edge") if needs_pad else arr
    return padded, tuple(factors)


def _reshape_into_blocks(
    padded: np.ndarray[Any, Any],
    factors: Sequence[int],
) -> tuple[np.ndarray[Any, Any], tuple[int, ...]]:
    split_shape: list[int] = []
    coarse_shape: list[int] = []
    non_block_axes: list[int] = []
    block_axes: list[int] = []
    for dim, (length, fac) in enumerate(zip(padded.shape, factors, strict=True)):
        coarse_len = length // fac
        coarse_shape.append(coarse_len)
        split_shape.extend([coarse_len, fac])
        non_block_axes.append(2 * dim)
        block_axes.append(2 * dim + 1)
    reshaped = padded.reshape(split_shape)
    transposed = reshaped.transpose(non_block_axes + block_axes)
    block_size = int(np.prod(factors))
    return transposed.reshape((*coarse_shape, block_size)), tuple(coarse_shape)


def downsample_block_mean(
    arr: np.ndarray[Any, Any],
    axes: Sequence[AxisSpec],
) -> tuple[np.ndarray[Any, Any], tuple[AxisSpec, ...]]:
    """Downsample spatial axes by 2x using block averaging, preserving floating precision."""
    spatial_flags = [ax.kind == "space" for ax in axes]
    padded, factors = _pad_for_factor2(arr, spatial_flags)
    if all(f == 1 for f in factors):
        return arr.copy(), tuple(axes)
    blocks, _ = _reshape_into_blocks(padded, factors)
    out_dtype = np.float64 if arr.dtype == np.float64 else np.float32
    mean_arr: np.ndarray[Any, Any] = np.asarray(
        np.mean(blocks, axis=-1, dtype=np.float64), dtype=out_dtype
    )
    return mean_arr, coarsen_axes(axes, factors)


def downsample_block_mode(
    arr: np.ndarray[Any, Any],
    axes: Sequence[AxisSpec],
) -> tuple[np.ndarray[Any, Any], tuple[AxisSpec, ...]]:
    """Downsample integer label spatial axes by 2x using exact mode (smallest ID breaks ties)."""
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"downsample_block_mode requires an integer array, got dtype={arr.dtype}.")
    spatial_flags = [ax.kind == "space" for ax in axes]
    padded, factors = _pad_for_factor2(arr, spatial_flags)
    if all(f == 1 for f in factors):
        return arr.copy(), tuple(axes)
    blocks, _ = _reshape_into_blocks(padded, factors)
    sorted_blocks = np.sort(blocks, axis=-1)
    block_size = sorted_blocks.shape[-1]
    counts = np.stack(
        [
            np.sum(sorted_blocks == sorted_blocks[..., j : j + 1], axis=-1)
            for j in range(block_size)
        ],
        axis=-1,
    )
    best_idx = np.argmax(counts, axis=-1)
    mode_arr = np.take_along_axis(sorted_blocks, best_idx[..., None], axis=-1)[..., 0].astype(
        arr.dtype, copy=False
    )
    return mode_arr, coarsen_axes(axes, factors)


def _axes_to_ngff_v04(axes: Sequence[AxisSpec]) -> list[dict[str, Any]]:
    ngff_axes: list[dict[str, Any]] = []
    for ax in axes:
        if ax.kind == "space":
            entry: dict[str, Any] = {"name": ax.name, "type": "space"}
            if ax.unit in _CANONICAL_TO_NGFF_SPACE_UNIT:
                entry["unit"] = _CANONICAL_TO_NGFF_SPACE_UNIT[ax.unit]
            ngff_axes.append(entry)
        elif ax.kind == "time":
            entry = {"name": ax.name, "type": "time"}
            if ax.unit in _CANONICAL_TO_NGFF_TIME_UNIT:
                entry["unit"] = _CANONICAL_TO_NGFF_TIME_UNIT[ax.unit]
            ngff_axes.append(entry)
        elif ax.kind == "channel":
            ngff_axes.append({"name": ax.name, "type": "channel"})
        else:
            ngff_axes.append({"name": ax.name, "type": ax.kind})
    return ngff_axes


def _axes_to_basecoat_json(axes: Sequence[AxisSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": ax.name,
            "kind": ax.kind,
            "unit": ax.unit,
            "scale": float(ax.scale),
            "offset": float(ax.offset),
        }
        for ax in axes
    ]


def _default_chunks(shape: Sequence[int], axes: Sequence[AxisSpec]) -> tuple[int, ...]:
    return tuple(
        min(int(dim), 1 if ax.kind == "channel" else 128)
        for dim, ax in zip(shape, axes, strict=True)
    )


def _write_multiscale_group(
    group: Any,
    vol: Volume,
    *,
    name: str,
    multiscale_levels: int,
    chunks: Sequence[int] | None,
    is_label: bool,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    if multiscale_levels < 1:
        raise ValueError(f"multiscale_levels must be >= 1, got {multiscale_levels}.")
    compressor = numcodecs.Blosc(
        cname="zstd",
        clevel=5,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    datasets_meta: list[dict[str, Any]] = []
    level_axes_json: dict[str, list[dict[str, Any]]] = {}
    cur_arr = vol.data
    cur_axes = vol.axes

    for lvl in range(multiscale_levels):
        lvl_key = str(lvl)
        if chunks is not None and lvl == 0:
            if len(chunks) != cur_arr.ndim:
                raise ValueError(
                    f"chunks length ({len(chunks)}) must match volume ndim ({cur_arr.ndim})."
                )
            lvl_chunks = tuple(
                min(int(s), max(1, int(c))) for s, c in zip(cur_arr.shape, chunks, strict=True)
            )
        else:
            lvl_chunks = _default_chunks(cur_arr.shape, cur_axes)

        group.create_dataset(
            lvl_key,
            data=cur_arr,
            chunks=lvl_chunks,
            compressor=compressor,
            overwrite=True,
        )

        transforms_list: list[dict[str, Any]] = [
            {"type": "scale", "scale": [float(ax.scale) for ax in cur_axes]}
        ]
        if any(ax.offset != 0.0 for ax in cur_axes):
            transforms_list.append(
                {"type": "translation", "translation": [float(ax.offset) for ax in cur_axes]}
            )

        datasets_meta.append({"path": lvl_key, "coordinateTransformations": transforms_list})
        level_axes_json[lvl_key] = _axes_to_basecoat_json(cur_axes)

        if lvl + 1 < multiscale_levels:
            if all(
                (ax.kind != "space" or s <= 1)
                for s, ax in zip(cur_arr.shape, cur_axes, strict=True)
            ):
                break
            if is_label:
                cur_arr, cur_axes = downsample_block_mode(cur_arr, cur_axes)
            else:
                cur_arr, cur_axes = downsample_block_mean(cur_arr, cur_axes)

    multiscale_entry: dict[str, Any] = {
        "version": "0.4",
        "name": name,
        "axes": _axes_to_ngff_v04(vol.axes),
        "datasets": datasets_meta,
        "type": "block_mode" if is_label else "block_mean",
    }

    attrs: dict[str, Any] = {
        "multiscales": [multiscale_entry],
        "basecoat_axes_by_level": level_axes_json,
        "basecoat_centering": vol.centering,
    }
    if vol.space is not None:
        attrs["basecoat_space"] = vol.space
    if provenance is not None:
        attrs["basecoat_provenance"] = dict(provenance)

    if is_label:
        unique_ids = [int(v) for v in np.unique(vol.data)]
        lbl_names = vol.label_names or {}
        all_ids = sorted(set(unique_ids) | set(lbl_names.keys()))
        colors = [
            {
                "label-value": lid,
                "rgba": list(_DEFAULT_LABEL_PALETTE_RGBA[idx % len(_DEFAULT_LABEL_PALETTE_RGBA)]),
            }
            for idx, lid in enumerate(all_ids)
        ]
        image_label_meta: dict[str, Any] = {
            "version": "0.4",
            "colors": colors,
            "source": {"image": "../../"},
        }
        if lbl_names:
            image_label_meta["properties"] = [
                {"label-value": int(lid), "name": str(lbl_names[lid])} for lid in sorted(lbl_names)
            ]
        attrs["image-label"] = image_label_meta

    group.attrs.update(attrs)


def write_ome_zarr(
    source: Volume | VolumeSet,
    path: str | Path,
    *,
    name: str | None = None,
    multiscale_levels: int = 2,
    chunks: Sequence[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a :class:`Volume` or co-registered :class:`VolumeSet` to an OME-Zarr v0.4 directory store."""
    out = Path(path)
    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"OME-Zarr target '{out}' already exists; pass overwrite=True to replace it."
            )
        shutil.rmtree(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    img_vol = source.image if isinstance(source, VolumeSet) else source
    labels_dict = source.labels if isinstance(source, VolumeSet) else {}
    unc_dict = source.uncertainties if isinstance(source, VolumeSet) else {}

    is_int_single_label = (not isinstance(source, VolumeSet)) and (img_vol.label_names is not None)
    stamped_prov = stamp_provenance(img_vol.provenance)

    store = zarr.DirectoryStore(str(out), dimension_separator="/")
    root = zarr.group(store=store, overwrite=True)

    dataset_name = name or out.stem.removesuffix(".ome")
    _write_multiscale_group(
        root,
        img_vol,
        name=dataset_name,
        multiscale_levels=multiscale_levels,
        chunks=chunks,
        is_label=is_int_single_label,
        provenance=stamped_prov,
    )

    actual_levels = len(root.attrs["multiscales"][0]["datasets"])

    if labels_dict:
        labels_grp = root.create_group("labels")
        label_keys = sorted(labels_dict.keys())
        labels_grp.attrs["labels"] = label_keys
        for key in label_keys:
            lvol = labels_dict[key]
            sub_grp = labels_grp.create_group(key)
            _write_multiscale_group(
                sub_grp,
                lvol,
                name=key,
                multiscale_levels=actual_levels,
                chunks=None,
                is_label=True,
                provenance=stamp_provenance(lvol.provenance),
            )

    if unc_dict:
        unc_grp = root.create_group("uncertainties")
        unc_keys = sorted(unc_dict.keys())
        unc_grp.attrs["uncertainties"] = unc_keys
        for key in unc_keys:
            uvol = unc_dict[key]
            sub_grp = unc_grp.create_group(key)
            _write_multiscale_group(
                sub_grp,
                uvol,
                name=key,
                multiscale_levels=actual_levels,
                chunks=None,
                is_label=False,
                provenance=stamp_provenance(uvol.provenance),
            )

    return out


def _parse_group_metadata(
    group: Any,
    level: int,
) -> tuple[
    str,
    tuple[AxisSpec, ...],
    str | None,
    Centering,
    dict[int, str] | None,
    dict[str, Any],
]:
    attrs = dict(group.attrs)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("Zarr group is missing valid OME-NGFF v0.4 'multiscales' metadata.")

    ms0 = multiscales[0]
    datasets = ms0.get("datasets", [])
    if level < 0 or level >= len(datasets):
        raise IndexError(
            f"Requested pyramid level={level}, but OME-Zarr store has {len(datasets)} level(s) (0..{len(datasets) - 1})."
        )

    ds_entry = datasets[level]
    ds_path = str(ds_entry["path"])

    saved_axes_by_level = attrs.get("basecoat_axes_by_level")
    if isinstance(saved_axes_by_level, Mapping) and ds_path in saved_axes_by_level:
        axes = tuple(
            AxisSpec(
                name=str(item["name"]),
                kind=item["kind"],
                unit=str(item["unit"]),
                scale=float(item["scale"]),
                offset=float(item["offset"]),
            )
            for item in saved_axes_by_level[ds_path]
        )
    else:
        raw_axes = ms0.get("axes", [])
        transforms = ds_entry.get("coordinateTransformations", [])
        scales = [1.0] * len(raw_axes)
        offsets = [0.0] * len(raw_axes)
        for tfm in transforms:
            if tfm.get("type") == "scale":
                scales = [float(s) for s in tfm["scale"]]
            elif tfm.get("type") == "translation":
                offsets = [float(t) for t in tfm["translation"]]

        parsed_axes: list[AxisSpec] = []
        for raw_ax, sc, of in zip(raw_axes, scales, offsets, strict=True):
            ax_type = str(raw_ax.get("type", "space"))
            kind: AxisKind = (
                cast(AxisKind, ax_type)
                if ax_type in ("space", "channel", "time", "angle", "spectral", "index")
                else "space"
            )
            raw_unit = raw_ax.get("unit")
            if raw_unit is not None:
                unit = _NGFF_TO_CANONICAL_UNIT.get(str(raw_unit), str(raw_unit))
            else:
                unit = "none" if kind == "channel" else "voxel"
            parsed_axes.append(
                AxisSpec(
                    name=str(raw_ax["name"]),
                    kind=kind,
                    unit=unit,
                    scale=sc,
                    offset=of,
                )
            )
        axes = tuple(parsed_axes)

    space_val = attrs.get("basecoat_space")
    space = str(space_val) if isinstance(space_val, str) else None

    centering_val = attrs.get("basecoat_centering", "cell")
    centering: Centering = (
        centering_val if centering_val in ("cell", "node", "face_x", "face_y", "face_z") else "cell"
    )

    label_names: dict[int, str] | None = None
    img_lbl = attrs.get("image-label")
    if isinstance(img_lbl, Mapping):
        props = img_lbl.get("properties")
        if isinstance(props, list):
            extracted: dict[int, str] = {}
            for p in props:
                if isinstance(p, Mapping) and "label-value" in p and "name" in p:
                    extracted[int(p["label-value"])] = str(p["name"])
            if extracted:
                label_names = extracted

    prov_raw = attrs.get("basecoat_provenance")
    prov = dict(prov_raw) if isinstance(prov_raw, Mapping) else {}
    return ds_path, axes, space, centering, label_names, prov


def open_ome_zarr(
    path: str | Path,
    *,
    level: int = 0,
    label: str | None = None,
    uncertainty: str | None = None,
) -> LazyVolume:
    """Open an OME-Zarr v0.4 array as a non-materialising :class:`LazyVolume`."""
    if label is not None and uncertainty is not None:
        raise ValueError("Pass at most one of label='...' or uncertainty='...', not both.")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OME-Zarr store does not exist: '{p}'.")

    root = zarr.open_group(str(p), mode="r")
    if label is not None:
        if "labels" not in root or label not in root["labels"]:
            available = list(root["labels"].group_keys()) if "labels" in root else []
            raise KeyError(f"Label '{label}' not found in '{p}/labels' (available: {available}).")
        target_grp = root["labels"][label]
    elif uncertainty is not None:
        if "uncertainties" not in root or uncertainty not in root["uncertainties"]:
            available = list(root["uncertainties"].group_keys()) if "uncertainties" in root else []
            raise KeyError(
                f"Uncertainty '{uncertainty}' not found in '{p}/uncertainties' (available: {available})."
            )
        target_grp = root["uncertainties"][uncertainty]
    else:
        target_grp = root

    ds_path, axes, space, centering, label_names, prov = _parse_group_metadata(target_grp, level)
    z_arr = target_grp[ds_path]

    def _read_slices(slices: tuple[slice, ...]) -> np.ndarray[Any, Any]:
        return np.asarray(z_arr[slices])

    return LazyVolume(
        shape=tuple(int(s) for s in z_arr.shape),
        dtype=np.dtype(z_arr.dtype),
        axes=axes,
        reader=_read_slices,
        closer=None,
        space=space,
        centering=centering,
        label_names=label_names,
        provenance=prov,
    )


@overload
def read_ome_zarr(
    path: str | Path,
    *,
    level: int = 0,
    label: str | None = None,
    uncertainty: str | None = None,
    as_set: Literal[False] = False,
) -> Volume: ...


@overload
def read_ome_zarr(
    path: str | Path,
    *,
    level: int = 0,
    label: None = None,
    uncertainty: None = None,
    as_set: Literal[True],
) -> VolumeSet: ...


def read_ome_zarr(
    path: str | Path,
    *,
    level: int = 0,
    label: str | None = None,
    uncertainty: str | None = None,
    as_set: bool = False,
) -> Volume | VolumeSet:
    """Read an OME-Zarr v0.4 store into a :class:`Volume` or co-registered :class:`VolumeSet`."""
    if as_set and (label is not None or uncertainty is not None):
        raise ValueError(
            "Pass either label='...' / uncertainty='...' (single Volume) or as_set=True, not both."
        )

    p = Path(path)
    if not as_set:
        with open_ome_zarr(p, level=level, label=label, uncertainty=uncertainty) as lv:
            return lv.to_volume(max_bytes=1024 * 1024**3)

    img_vol = read_ome_zarr(p, level=level, as_set=False)
    root = zarr.open_group(str(p), mode="r")
    labels_out: dict[str, Volume] = {}
    if "labels" in root:
        lbl_grp = root["labels"]
        declared = lbl_grp.attrs.get("labels")
        keys = (
            [str(k) for k in declared]
            if isinstance(declared, list)
            else sorted(lbl_grp.group_keys())
        )
        for key in keys:
            if key in lbl_grp:
                labels_out[key] = read_ome_zarr(p, level=level, label=key, as_set=False)

    unc_out: dict[str, Volume] = {}
    if "uncertainties" in root:
        unc_grp = root["uncertainties"]
        declared_unc = unc_grp.attrs.get("uncertainties")
        u_keys = (
            [str(k) for k in declared_unc]
            if isinstance(declared_unc, list)
            else sorted(unc_grp.group_keys())
        )
        for key in u_keys:
            if key in unc_grp:
                unc_out[key] = read_ome_zarr(p, level=level, uncertainty=key, as_set=False)

    return VolumeSet(img_vol, labels=labels_out, uncertainties=unc_out)
