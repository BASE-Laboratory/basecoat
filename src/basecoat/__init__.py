"""basecoat: Core volume container, region-lazy I/O, and coordinate transforms."""

from __future__ import annotations

from ._provenance import __version__
from .data import fetch_file
from .exceptions import (
    BasecoatError,
    ChecksumError,
    SpaceMismatchError,
    UnitConversionError,
)
from .geometry import AffineTransform3D, SimilarityTransform3D, resample_volume
from .io import (
    open_volume,
    read_ome_zarr,
    read_volume,
    write_ome_zarr,
    write_volume,
)
from .volume import (
    AxisSpec,
    LazyVolume,
    Volume,
    VolumeProtocol,
    VolumeSet,
    coarsen_axes,
)

__all__ = [
    "__version__",
    "BasecoatError",
    "ChecksumError",
    "SpaceMismatchError",
    "UnitConversionError",
    "AxisSpec",
    "LazyVolume",
    "Volume",
    "VolumeProtocol",
    "VolumeSet",
    "coarsen_axes",
    "open_volume",
    "read_ome_zarr",
    "read_volume",
    "write_ome_zarr",
    "write_volume",
    "AffineTransform3D",
    "SimilarityTransform3D",
    "fetch_file",
    "resample_volume",
]
