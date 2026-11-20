"""Core volume containers, AxisSpec, LazyVolume, and minimal VolumeProtocol.

Coordinate convention
---------------------
``AxisSpec.offset`` is the physical coordinate of the **cell centre** at index ``0``:

    x(i) = offset + i * scale

Consequently, a volume of length ``N`` along an axis spans outer voxel faces
``[offset - 0.5 * scale, offset + (N - 0.5) * scale]``. When block-downsampling a
spatial axis by factor ``k`` while keeping the outer physical domain walls fixed,
``coarsen_axes`` shifts the coarse grid's index-``0`` cell centre to:

    offset_coarse = offset + 0.5 * (k - 1) * scale

Portions of the cell-centred grid algebra build on workflows developed by
James Le Houx and Calum Green in ``MultimodalBenchmark`` and ``OpenLSR-X``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .exceptions import SpaceMismatchError, UnitConversionError

AxisKind = Literal["space", "channel", "time", "angle", "spectral", "index"]
Centering = Literal["cell", "node", "face_x", "face_y", "face_z"]

_UNIT_ALIASES: dict[str, str] = {
    "micrometer": "um",
    "micrometre": "um",
    "micron": "um",
    "µm": "um",
    "μm": "um",
    "nanometer": "nm",
    "nanometre": "nm",
    "millimeter": "mm",
    "millimetre": "mm",
    "centimeter": "cm",
    "centimetre": "cm",
    "meter": "m",
    "metre": "m",
    "ångström": "angstrom",
    "Å": "angstrom",
    "degree": "deg",
    "degrees": "deg",
    "radian": "rad",
    "radians": "rad",
    "second": "s",
    "millisecond": "ms",
}

_LENGTH_TO_METRES: dict[str, float] = {
    "angstrom": 1e-10,
    "nm": 1e-9,
    "um": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
}


def canonicalize_unit(unit: str) -> str:
    """Normalise common unit spellings to canonical short names."""
    cleaned = unit.strip()
    if not cleaned:
        raise ValueError("Axis unit must be a non-empty string (use 'voxel' when uncalibrated).")
    return _UNIT_ALIASES.get(cleaned, _UNIT_ALIASES.get(cleaned.lower(), cleaned))


def convert_length(value: float, from_unit: str, to_unit: str, *, axis_name: str = "?") -> float:
    """Convert a spatial length between SI/metric units, failing loudly on 'voxel'."""
    src = canonicalize_unit(from_unit)
    dst = canonicalize_unit(to_unit)
    if src == dst and src in _LENGTH_TO_METRES:
        return float(value)
    if src == "voxel":
        raise UnitConversionError(
            f"Volume axis '{axis_name}' is uncalibrated (unit='voxel') and cannot be converted "
            f"to '{dst}'. Supply physical spacing via Volume.from_array(arr, voxel_size_um=...) "
            f"or vol.with_voxel_size_um(...)."
        )
    if src not in _LENGTH_TO_METRES:
        raise UnitConversionError(
            f"Axis '{axis_name}' has unit='{from_unit}', which is not a supported physical length "
            f"unit ({sorted(_LENGTH_TO_METRES)})."
        )
    if dst not in _LENGTH_TO_METRES:
        raise UnitConversionError(
            f"Target unit '{to_unit}' is not a supported physical length unit "
            f"({sorted(_LENGTH_TO_METRES)})."
        )
    return float(value) * (_LENGTH_TO_METRES[src] / _LENGTH_TO_METRES[dst])


@dataclass(frozen=True)
class AxisSpec:
    """Specification for one array axis under the index-0 cell-centre convention.

    Parameters
    ----------
    name : str
        Axis identifier (e.g. ``'z'``, ``'y'``, ``'x'``, ``'c'``).
    kind : AxisKind
        Semantic axis kind (default ``'space'``).
    unit : str
        Physical unit string (e.g. ``'um'``, ``'m'``, ``'deg'``) or ``'voxel'`` when uncalibrated.
    scale : float
        Physical increment per unit index step. Must be finite and strictly positive for
        ``'space'`` and ``'time'`` axes.
    offset : float
        Physical coordinate of the **cell centre** at index ``0``, so index ``i`` sits at
        ``offset + i * scale``.
    """

    name: str
    kind: AxisKind = "space"
    unit: str = "voxel"
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("AxisSpec.name must be a non-empty string.")
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "unit", canonicalize_unit(self.unit))

        sc = float(self.scale)
        of = float(self.offset)
        if not math.isfinite(sc) or not math.isfinite(of):
            raise ValueError(
                f"AxisSpec('{clean_name}') scale and offset must be finite, got scale={self.scale!r}, offset={self.offset!r}."
            )
        if self.kind in ("space", "time") and sc <= 0.0:
            raise ValueError(
                f"AxisSpec('{clean_name}') of kind='{self.kind}' requires scale > 0, got {sc!r}."
            )
        if sc == 0.0:
            raise ValueError(f"AxisSpec('{clean_name}') scale must be non-zero.")
        object.__setattr__(self, "scale", sc)
        object.__setattr__(self, "offset", of)

    def index_to_world(self, index: float | np.ndarray) -> float | np.ndarray:
        """Map continuous or integer index ``i`` to physical cell-centre coordinate ``offset + i * scale``."""
        if isinstance(index, np.ndarray):
            return self.offset + np.asarray(index, dtype=np.float64) * self.scale
        return float(self.offset + float(index) * self.scale)

    def world_to_index(self, coord: float | np.ndarray) -> float | np.ndarray:
        """Map physical coordinate ``x`` to continuous index ``(x - offset) / scale``."""
        if isinstance(coord, np.ndarray):
            return (np.asarray(coord, dtype=np.float64) - self.offset) / self.scale
        return float((float(coord) - self.offset) / self.scale)

    def scale_in(self, unit: str) -> float:
        """Return ``scale`` converted to *unit*."""
        return convert_length(self.scale, self.unit, unit, axis_name=self.name)

    def offset_in(self, unit: str) -> float:
        """Return ``offset`` converted to *unit*."""
        return convert_length(self.offset, self.unit, unit, axis_name=self.name)

    def sliced(self, slc: slice, length: int) -> AxisSpec:
        """Return updated ``AxisSpec`` after applying a positive-step ``slice`` to an axis of *length*."""
        start, stop, step = slc.indices(length)
        if step <= 0:
            raise ValueError(
                f"Negative slice steps are not supported on Volume axis '{self.name}' (got {slc!r}); "
                f"use positive slice steps so axis orientation and positive scale are preserved."
            )
        if stop <= start:
            raise ValueError(
                f"Empty slice {slc!r} on axis '{self.name}' of length {length} produces zero-length dimension."
            )
        return AxisSpec(
            name=self.name,
            kind=self.kind,
            unit=self.unit,
            scale=self.scale * step,
            offset=self.offset + start * self.scale,
        )


def coarsen_axes(
    axes: Sequence[AxisSpec],
    factors: int | Sequence[int],
) -> tuple[AxisSpec, ...]:
    """Compute coarsened ``AxisSpec`` tuple preserving outer physical voxel boundaries.

    When an integer ``factors`` is given, ``'space'`` axes are coarsened by ``factors`` and
    non-spatial axes (e.g. ``'channel'``) use factor ``1``. When a sequence is given, its
    length must match ``len(axes)``.
    """
    if isinstance(factors, int):
        if factors < 1:
            raise ValueError(f"Coarsening factor must be >= 1, got {factors}.")
        fac_tuple = tuple(factors if ax.kind == "space" else 1 for ax in axes)
    else:
        if len(factors) != len(axes):
            raise ValueError(
                f"factors length ({len(factors)}) must match axes length ({len(axes)})."
            )
        fac_tuple = tuple(int(k) for k in factors)
        if any(k < 1 for k in fac_tuple):
            raise ValueError(f"All coarsening factors must be >= 1, got {fac_tuple}.")

    out: list[AxisSpec] = []
    for ax, k in zip(axes, fac_tuple, strict=True):
        if k == 1:
            out.append(ax)
        else:
            out.append(
                AxisSpec(
                    name=ax.name,
                    kind=ax.kind,
                    unit=ax.unit,
                    scale=ax.scale * k,
                    offset=ax.offset + 0.5 * (k - 1) * ax.scale,
                )
            )
    return tuple(out)


def _default_axes_names(ndim: int) -> tuple[str, ...]:
    if ndim == 2:
        return ("y", "x")
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 4:
        return ("c", "z", "y", "x")
    raise ValueError(
        f"Cannot infer default axis names for {ndim}-D array; pass explicit axes= or axes_names=."
    )


def _normalize_slice_tuple(
    key: slice | tuple[slice | Any, ...],
    ndim: int,
) -> tuple[slice, ...]:
    items: tuple[Any, ...] = (key,) if not isinstance(key, tuple) else key
    expanded: list[Any] = []
    ellipsis_seen = False
    n_explicit = sum(1 for item in items if item is not Ellipsis)
    if n_explicit > ndim:
        raise IndexError(f"Too many slice indices ({n_explicit}) for {ndim}-D volume.")
    for item in items:
        if item is Ellipsis:
            if ellipsis_seen:
                raise IndexError("An index can only have a single ellipsis ('...').")
            ellipsis_seen = True
            expanded.extend([slice(None)] * (ndim - n_explicit))
        else:
            expanded.append(item)
    while len(expanded) < ndim:
        expanded.append(slice(None))

    out: list[slice] = []
    for dim, item in enumerate(expanded):
        if not isinstance(item, slice):
            raise TypeError(
                f"Volume indexing at dimension {dim} received {type(item).__name__} ({item!r}); "
                f"use slice syntax (e.g. vol[i:i+1, :, :]) so all {ndim} axes stay explicit."
            )
        out.append(item)
    return tuple(out)


@runtime_checkable
class VolumeProtocol(Protocol):
    """Minimal duck-typed protocol consumed by Layer 3 solvers without importing basecoat."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def axis_names(self) -> tuple[str, ...]: ...

    def spacing_in(self, unit: str = "m") -> tuple[float, ...]: ...

    def __array__(
        self,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, Any]: ...


class Volume:
    """Single-array volumetric container paired with explicit per-axis physical coordinates.

    ``Volume`` holds exactly ONE ``numpy.ndarray`` plus its ``AxisSpec`` metadata. Co-registered
    label masks live alongside an image inside :class:`VolumeSet`.
    """

    def __init__(
        self,
        data: np.ndarray[Any, Any],
        axes: Sequence[AxisSpec],
        *,
        space: str | None = None,
        centering: Centering = "cell",
        label_names: Mapping[int, str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(data)
        if arr.ndim < 2:
            raise ValueError(f"Volume requires at least a 2-D array, got shape={arr.shape}.")
        axes_tuple = tuple(axes)
        if len(axes_tuple) != arr.ndim:
            raise ValueError(
                f"Number of axes ({len(axes_tuple)}) must match array ndim ({arr.ndim}, shape={arr.shape})."
            )
        names = [ax.name for ax in axes_tuple]
        if len(set(names)) != len(names):
            raise ValueError(f"Volume axis names must be unique, got {names}.")
        if centering not in ("cell", "node", "face_x", "face_y", "face_z"):
            raise ValueError(f"Unsupported centering={centering!r}.")

        lbl_dict: dict[int, str] | None = None
        if label_names is not None:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    f"label_names can only be attached to an integer label Volume, got dtype={arr.dtype}."
                )
            lbl_dict = {int(k): str(v) for k, v in label_names.items()}

        self._data: np.ndarray[Any, Any] = arr
        self._axes: tuple[AxisSpec, ...] = axes_tuple
        self._space: str | None = (
            space.strip() if isinstance(space, str) and space.strip() else None
        )
        self._centering: Centering = centering
        self._label_names: dict[int, str] | None = lbl_dict
        self._provenance: dict[str, Any] = dict(provenance) if provenance is not None else {}

    @classmethod
    def from_array(
        cls,
        data: np.ndarray[Any, Any],
        *,
        voxel_size_um: float | Sequence[float] | None = None,
        origin_um: Sequence[float] | None = None,
        axes_names: Sequence[str] | None = None,
        space: str | None = None,
        centering: Centering = "cell",
        label_names: Mapping[int, str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Volume:
        """Construct a ``Volume`` from a NumPy array with optional physical spacing in ``'um'``.

        When *voxel_size_um* is ``None``, spatial axes are marked explicitly as ``unit='voxel'``.
        """
        arr = np.asarray(data)
        names = tuple(axes_names) if axes_names is not None else _default_axes_names(arr.ndim)
        if len(names) != arr.ndim:
            raise ValueError(
                f"axes_names length ({len(names)}) must match array ndim ({arr.ndim})."
            )
        spatial_indices = [i for i, n in enumerate(names) if n != "c"]
        n_space = len(spatial_indices)

        if voxel_size_um is None:
            if origin_um is not None:
                raise ValueError(
                    "origin_um cannot be set without voxel_size_um; pass voxel_size_um=... first."
                )
            axes = [
                AxisSpec(name=n, kind="channel", unit="none", scale=1.0, offset=0.0)
                if n == "c"
                else AxisSpec(name=n, kind="space", unit="voxel", scale=1.0, offset=0.0)
                for n in names
            ]
        else:
            if isinstance(voxel_size_um, (int, float, np.integer, np.floating)):
                scales = [float(voxel_size_um)] * n_space
            else:
                scales = [float(v) for v in voxel_size_um]
                if len(scales) != n_space:
                    raise ValueError(
                        f"voxel_size_um length ({len(scales)}) must match number of spatial axes ({n_space})."
                    )
            if origin_um is None:
                offsets = [0.0] * n_space
            else:
                offsets = [float(o) for o in origin_um]
                if len(offsets) != n_space:
                    raise ValueError(
                        f"origin_um length ({len(offsets)}) must match number of spatial axes ({n_space})."
                    )
            axes = []
            sp_cursor = 0
            for n in names:
                if n == "c":
                    axes.append(
                        AxisSpec(name="c", kind="channel", unit="none", scale=1.0, offset=0.0)
                    )
                else:
                    axes.append(
                        AxisSpec(
                            name=n,
                            kind="space",
                            unit="um",
                            scale=scales[sp_cursor],
                            offset=offsets[sp_cursor],
                        )
                    )
                    sp_cursor += 1

        return cls(
            arr,
            axes,
            space=space,
            centering=centering,
            label_names=label_names,
            provenance=provenance,
        )

    @classmethod
    def uncalibrated(
        cls,
        data: np.ndarray[Any, Any],
        *,
        axes_names: Sequence[str] | None = None,
        label_names: Mapping[int, str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Volume:
        """Construct an explicitly uncalibrated ``unit='voxel'`` volume."""
        return cls.from_array(
            data,
            voxel_size_um=None,
            axes_names=axes_names,
            label_names=label_names,
            provenance=provenance,
        )

    @property
    def data(self) -> np.ndarray[Any, Any]:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._data.shape)

    @property
    def ndim(self) -> int:
        return int(self._data.ndim)

    @property
    def dtype(self) -> np.dtype[Any]:
        dt: np.dtype[Any] = self._data.dtype
        return dt

    @property
    def nbytes(self) -> int:
        return int(self._data.nbytes)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return self._axes

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(ax.name for ax in self._axes)

    @property
    def space(self) -> str | None:
        return self._space

    @property
    def centering(self) -> Centering:
        return self._centering

    @property
    def label_names(self) -> dict[int, str] | None:
        return dict(self._label_names) if self._label_names is not None else None

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    @property
    def is_calibrated(self) -> bool:
        """True if every spatial axis carries a physical length unit rather than 'voxel'."""
        space_axes = [ax for ax in self._axes if ax.kind == "space"]
        return bool(space_axes) and all(ax.unit in _LENGTH_TO_METRES for ax in space_axes)

    def spacing_in(self, unit: str = "m") -> tuple[float, ...]:
        """Return physical spacing of all ``'space'`` axes in *unit* (ordered as in ``axes``)."""
        space_axes = [ax for ax in self._axes if ax.kind == "space"]
        if not space_axes:
            raise UnitConversionError("Volume has no spatial axes.")
        return tuple(ax.scale_in(unit) for ax in space_axes)

    def origin_in(self, unit: str = "m") -> tuple[float, ...]:
        """Return index-0 cell-centre physical coordinates of all ``'space'`` axes in *unit*."""
        space_axes = [ax for ax in self._axes if ax.kind == "space"]
        if not space_axes:
            raise UnitConversionError("Volume has no spatial axes.")
        return tuple(ax.offset_in(unit) for ax in space_axes)

    def with_voxel_size_um(
        self,
        voxel_size_um: float | Sequence[float],
        *,
        origin_um: Sequence[float] | None = None,
        space: str | None = None,
    ) -> Volume:
        """Return a new ``Volume`` sharing the underlying array with calibrated ``'um'`` spatial axes."""
        space_axes = [ax for ax in self._axes if ax.kind == "space"]
        n_space = len(space_axes)
        if isinstance(voxel_size_um, (int, float, np.integer, np.floating)):
            scales = [float(voxel_size_um)] * n_space
        else:
            scales = [float(v) for v in voxel_size_um]
            if len(scales) != n_space:
                raise ValueError(
                    f"voxel_size_um length ({len(scales)}) must match spatial axis count ({n_space})."
                )
        if origin_um is None:
            offsets = [0.0] * n_space
        else:
            offsets = [float(o) for o in origin_um]
            if len(offsets) != n_space:
                raise ValueError(
                    f"origin_um length ({len(offsets)}) must match spatial axis count ({n_space})."
                )

        new_axes: list[AxisSpec] = []
        cursor = 0
        for ax in self._axes:
            if ax.kind == "space":
                new_axes.append(
                    AxisSpec(
                        name=ax.name,
                        kind="space",
                        unit="um",
                        scale=scales[cursor],
                        offset=offsets[cursor],
                    )
                )
                cursor += 1
            else:
                new_axes.append(ax)

        return Volume(
            self._data,
            new_axes,
            space=space if space is not None else self._space,
            centering=self._centering,
            label_names=self._label_names,
            provenance=self._provenance,
        )

    def fortran_view(self) -> np.ndarray[Any, Any]:
        """Return a zero-copy Fortran-contiguous ``(x, y, z)`` view of a C-contiguous ``(z, y, x)`` volume."""
        if self.axis_names != ("z", "y", "x"):
            raise ValueError(
                f"fortran_view() requires 3-D ('z', 'y', 'x') axes, got {self.axis_names}."
            )
        if not self._data.flags.c_contiguous:
            raise ValueError(
                "fortran_view() requires a C-contiguous ('z', 'y', 'x') array to avoid copying; "
                "call np.ascontiguousarray(vol.data) explicitly if a copy is acceptable."
            )
        view: np.ndarray[Any, Any] = self._data.T
        return view

    def __getitem__(self, key: slice | tuple[slice | Any, ...]) -> Volume:
        slices = _normalize_slice_tuple(key, self.ndim)
        new_axes = tuple(
            ax.sliced(slc, dim_len)
            for ax, slc, dim_len in zip(self._axes, slices, self.shape, strict=True)
        )
        return Volume(
            self._data[slices],
            new_axes,
            space=self._space,
            centering=self._centering,
            label_names=self._label_names,
            provenance=self._provenance,
        )

    def __array__(
        self,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, Any]:
        if dtype is None and not copy:
            return self._data
        if copy:
            return np.array(self._data, dtype=dtype, copy=True)
        return np.asarray(self._data, dtype=dtype)

    def __repr__(self) -> str:
        axes_desc = ", ".join(f"{ax.name}:{ax.scale:g}{ax.unit}@{ax.offset:g}" for ax in self._axes)
        space_str = f", space={self._space!r}" if self._space else ""
        return f"Volume(shape={self.shape}, dtype={self.dtype}, axes=[{axes_desc}]{space_str})"


def _assert_coregistered(image: Volume, label_vol: Volume, label_key: str) -> None:
    if not np.issubdtype(label_vol.dtype, np.integer):
        raise ValueError(
            f"Label volume '{label_key}' in VolumeSet must have an integer dtype, got {label_vol.dtype}."
        )
    spatial_img = [
        (s, ax) for s, ax in zip(image.shape, image.axes, strict=True) if ax.kind == "space"
    ]
    if len(label_vol.axes) != len(spatial_img) or any(ax.kind != "space" for ax in label_vol.axes):
        raise SpaceMismatchError(
            f"Label volume '{label_key}' axes {label_vol.axis_names} must match spatial axes of image "
            f"{tuple(ax.name for _, ax in spatial_img)}."
        )
    for (s_img, ax_img), (s_lbl, ax_lbl) in zip(
        spatial_img,
        zip(label_vol.shape, label_vol.axes, strict=True),
        strict=True,
    ):
        if s_img != s_lbl or ax_img.name != ax_lbl.name or ax_img.unit != ax_lbl.unit:
            raise SpaceMismatchError(
                f"Label volume '{label_key}' grid (shape={label_vol.shape}, axes={label_vol.axes}) "
                f"does not match image spatial grid (shape={image.shape}, axes={image.axes})."
            )
        if not (
            math.isclose(ax_img.scale, ax_lbl.scale, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(ax_img.offset, ax_lbl.offset, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise SpaceMismatchError(
                f"Label volume '{label_key}' axis '{ax_lbl.name}' (scale={ax_lbl.scale}, offset={ax_lbl.offset}) "
                f"diverges from image axis (scale={ax_img.scale}, offset={ax_img.offset})."
            )
    if image.space is not None and label_vol.space is not None and image.space != label_vol.space:
        raise SpaceMismatchError(
            f"Label volume '{label_key}' space={label_vol.space!r} does not match image space={image.space!r}."
        )


class VolumeSet:
    """Co-registered container pairing a primary image ``Volume`` with named integer label ``Volume``s."""

    def __init__(
        self,
        image: Volume,
        labels: Mapping[str, Volume] | None = None,
    ) -> None:
        lbls: dict[str, Volume] = {}
        if labels is not None:
            for key, lv in labels.items():
                clean_key = key.strip()
                if not clean_key or "/" in clean_key or "\\" in clean_key:
                    raise ValueError(
                        f"Label key must be a plain identifier without slashes, got {key!r}."
                    )
                _assert_coregistered(image, lv, clean_key)
                lbls[clean_key] = lv
        self._image = image
        self._labels = lbls

    @property
    def image(self) -> Volume:
        return self._image

    @property
    def labels(self) -> dict[str, Volume]:
        return dict(self._labels)

    def __getitem__(self, key: slice | tuple[slice | Any, ...]) -> VolumeSet:
        slices = _normalize_slice_tuple(key, self._image.ndim)
        sliced_img = self._image[slices]
        spatial_slices = tuple(
            slc for slc, ax in zip(slices, self._image.axes, strict=True) if ax.kind == "space"
        )
        sliced_labels = {k: lv[spatial_slices] for k, lv in self._labels.items()}
        return VolumeSet(sliced_img, labels=sliced_labels)

    def __repr__(self) -> str:
        return f"VolumeSet(image={self._image!r}, labels={sorted(self._labels.keys())})"


class LazyVolume:
    """Out-of-core volumetric handle supporting region reads without silent RAM materialisation."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        axes: Sequence[AxisSpec],
        reader: Callable[[tuple[slice, ...]], np.ndarray[Any, Any]],
        closer: Callable[[], None] | None = None,
        space: str | None = None,
        centering: Centering = "cell",
        label_names: Mapping[int, str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        axes_tuple = tuple(axes)
        if len(shape) != len(axes_tuple):
            raise ValueError(
                f"LazyVolume shape length ({len(shape)}) must match axes length ({len(axes_tuple)})."
            )
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)
        self._axes = axes_tuple
        self._reader = reader
        self._closer = closer
        self._closed = False
        self._space = space
        self._centering: Centering = centering
        self._label_names = (
            {int(k): str(v) for k, v in label_names.items()} if label_names else None
        )
        self._provenance = dict(provenance) if provenance is not None else {}

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return int(math.prod(self._shape) * self._dtype.itemsize)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return self._axes

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(ax.name for ax in self._axes)

    @property
    def space(self) -> str | None:
        return self._space

    @property
    def centering(self) -> Centering:
        return self._centering

    @property
    def label_names(self) -> dict[int, str] | None:
        return dict(self._label_names) if self._label_names is not None else None

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    @property
    def closed(self) -> bool:
        return self._closed

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        raise TypeError(
            "LazyVolume refuses implicit NumPy materialisation to protect host memory; "
            "call lazy_vol.read_region((slice(z0, z1), ...)) for a sub-volume or "
            "lazy_vol.to_volume(max_bytes=...) to load explicitly."
        )

    def read_region(self, region: slice | tuple[slice | Any, ...]) -> Volume:
        """Read only the requested slice tuple into a calibrated :class:`Volume`."""
        if self._closed:
            raise ValueError("Cannot read from a closed LazyVolume.")
        slices = _normalize_slice_tuple(region, self.ndim)
        new_axes = tuple(
            ax.sliced(slc, dim_len)
            for ax, slc, dim_len in zip(self._axes, slices, self._shape, strict=True)
        )
        sub_arr = np.asarray(self._reader(slices))
        return Volume(
            sub_arr,
            new_axes,
            space=self._space,
            centering=self._centering,
            label_names=self._label_names,
            provenance=self._provenance,
        )

    def to_volume(self, *, max_bytes: int = 512 * 1024**2) -> Volume:
        """Explicitly load the entire volume into RAM if ``nbytes <= max_bytes``."""
        if self._closed:
            raise ValueError("Cannot read from a closed LazyVolume.")
        if self.nbytes > max_bytes:
            mib = self.nbytes / (1024.0**2)
            limit_mib = max_bytes / (1024.0**2)
            raise MemoryError(
                f"LazyVolume size is {mib:.1f} MiB, exceeding max_bytes={limit_mib:.1f} MiB. "
                f"Read a sub-region via lazy_vol.read_region(...) or pass a higher max_bytes= explicitly."
            )
        full_slices = tuple(slice(None) for _ in range(self.ndim))
        return self.read_region(full_slices)

    def close(self) -> None:
        """Close any underlying file handle."""
        if not self._closed:
            self._closed = True
            if self._closer is not None:
                self._closer()

    def __enter__(self) -> LazyVolume:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"LazyVolume(shape={self._shape}, dtype={self._dtype}, state={state})"
