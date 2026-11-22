"""basecoat: Core volume container, region-lazy I/O, and coordinate transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    open_ome_zarr,
    open_savu_h5,
    open_volume,
    read_ome_zarr,
    read_savu_subvolume,
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

if TYPE_CHECKING:
    from .filters import (
        ArtifactCleanupReport,
        compute_ring_anova_r2,
        detect_ring_center_yx,
        detrend_radial_cupping,
        remove_rings_polar_conservative,
    )
    from .layers import check_layer_imports

_LAZY_FILTERS = {
    "ArtifactCleanupReport",
    "compute_ring_anova_r2",
    "detect_ring_center_yx",
    "detrend_radial_cupping",
    "remove_rings_polar_conservative",
}


def __getattr__(name: str) -> object:
    if name == "check_layer_imports":
        from .layers import check_layer_imports

        return check_layer_imports
    if name in _LAZY_FILTERS:
        from . import filters as _filters

        return getattr(_filters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
