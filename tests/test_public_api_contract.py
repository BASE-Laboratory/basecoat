"""Locks basecoat's public top-level API (__all__) against silent changes."""

from __future__ import annotations

import basecoat

EXPECTED_V0_2_0_ALL = [
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
    "open_savu_h5",
    "open_volume",
    "read_ome_zarr",
    "read_savu_subvolume",
    "read_volume",
    "write_ome_zarr",
    "write_volume",
    "AffineTransform3D",
    "SimilarityTransform3D",
    "check_layer_imports",
    "fetch_file",
    "resample_volume",
]


def test_public_api_locked_to_version() -> None:
    assert basecoat.__version__ == "0.2.0", (
        "Update EXPECTED_V0_2_0_ALL intentionally when bumping basecoat.__version__."
    )
    assert list(basecoat.__all__) == EXPECTED_V0_2_0_ALL
    for sym in basecoat.__all__:
        assert hasattr(basecoat, sym), f"Exported symbol {sym!r} missing on basecoat module."
