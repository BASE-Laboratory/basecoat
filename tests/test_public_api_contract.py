"""Locks basecoat's public top-level API (__all__) against silent changes."""

from __future__ import annotations

import basecoat

EXPECTED_V0_3_0_ALL = [
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


def test_public_api_locked_to_version() -> None:
    assert basecoat.__version__ == "0.3.1", (
        "Update EXPECTED_V0_3_0_ALL intentionally when bumping basecoat.__version__."
    )
    assert list(basecoat.__all__) == EXPECTED_V0_3_0_ALL
    for sym in basecoat.__all__:
        assert hasattr(basecoat, sym), f"Exported symbol {sym!r} missing on basecoat module."
