"""Dev-only validation of hand-written OME-Zarr v0.4 stores using the community ome-zarr reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from basecoat import Volume, VolumeSet, write_ome_zarr


def test_handwritten_ome_zarr_passes_community_ome_zarr_reader(tmp_path: Path) -> None:
    ome_io = pytest.importorskip("ome_zarr.io")
    ome_reader = pytest.importorskip("ome_zarr.reader")

    img = Volume.from_array(
        np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16),
        voxel_size_um=(1.5, 0.75, 0.75),
        origin_um=(5.0, 10.0, 15.0),
    )
    lbl = Volume.from_array(
        (img.data > 1500).astype(np.int32),
        voxel_size_um=(1.5, 0.75, 0.75),
        origin_um=(5.0, 10.0, 15.0),
        label_names={0: "pore", 1: "solid"},
    )
    store_path = write_ome_zarr(
        VolumeSet(img, labels={"phase": lbl}),
        tmp_path / "validated.ome.zarr",
        multiscale_levels=2,
    )

    loc = ome_io.parse_url(str(store_path))
    assert loc is not None
    reader = ome_reader.Reader(loc)
    nodes = list(reader())
    assert len(nodes) >= 2  # Root Multiscales node + Labels node

    root_node = nodes[0]
    assert len(root_node.data) == 2
    assert tuple(root_node.data[0].shape) == (16, 16, 16)
    assert tuple(root_node.data[1].shape) == (8, 8, 8)
    assert [ax["name"] for ax in root_node.metadata["axes"]] == ["z", "y", "x"]
    assert [ax["unit"] for ax in root_node.metadata["axes"]] == [
        "micrometer",
        "micrometer",
        "micrometer",
    ]
