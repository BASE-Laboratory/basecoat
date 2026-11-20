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
from .filters import (
    ArtifactCleanupReport,
    compute_ring_anova_r2,
    detect_ring_center_yx,
    detrend_radial_cupping,
    remove_rings_polar_conservative,
)
from .geometry import AffineTransform3D, SimilarityTransform3D, resample_volume
from .io import (
    open_ome_zarr,
    open_savu_h5,
    open_volume,
    read_ome_zarr,
    read_savu_subvolume,
    read_volume,
    write_ome_zarr,
    write_volume,
)
from .layers import check_layer_imports
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
    "open_ome_zarr",
    "open_savu_h5",
    "open_volume",
    "read_ome_zarr",
    "read_savu_subvolume",
    "read_volume",
    "write_ome_zarr",
    "write_volume",
    "AffineTransform3D",
    "SimilarityTransform3D",
    "ArtifactCleanupReport",
    "check_layer_imports",
    "compute_ring_anova_r2",
    "detect_ring_center_yx",
    "detrend_radial_cupping",
    "fetch_file",
    "remove_rings_polar_conservative",
    "resample_volume",
]
