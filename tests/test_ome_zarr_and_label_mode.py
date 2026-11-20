"""Tests for OME-Zarr v0.4 multiscale storage, mode-downsampled labels, and true out-of-core chunk reads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basecoat import Volume, VolumeSet, __version__, open_volume, read_ome_zarr, write_ome_zarr


def test_label_pyramid_uses_mode_never_mean_and_preserves_cell_centre_translation(
    tmp_path: Path,
) -> None:
    img_arr = np.linspace(0.0, 100.0, 16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)

    # Labels use ONLY {0, 2, 7} (never 1 or 3!). A mean pyramid would invent 1s and 3s.
    lbl_arr = np.zeros((16, 16, 16), dtype=np.int32)
    # Block (0, 0, 0) [0:2, 0:2, 0:2]: five 7s and three 2s -> mode MUST be 7
    block_7 = np.array([7, 7, 7, 7, 7, 2, 2, 2], dtype=np.int32).reshape(2, 2, 2)
    lbl_arr[0:2, 0:2, 0:2] = block_7
    # Block (0, 0, 1) [0:2, 0:2, 2:4]: exact 4-vs-4 tie between 2 and 7 -> tie-break MUST pick smallest (2)
    block_tie = np.array([7, 7, 7, 7, 2, 2, 2, 2], dtype=np.int32).reshape(2, 2, 2)
    lbl_arr[0:2, 0:2, 2:4] = block_tie
    # Remainder alternating 0 and 2 with majority 2 in [2:4, 2:4, 2:4]
    lbl_arr[2:4, 2:4, 2:4] = np.array([2, 2, 2, 2, 2, 0, 0, 0], dtype=np.int32).reshape(2, 2, 2)

    img_vol = Volume.from_array(
        img_arr, voxel_size_um=0.8, origin_um=(10.0, 20.0, 30.0), space="zn13x_roi"
    )
    lbl_vol = Volume.from_array(
        lbl_arr,
        voxel_size_um=0.8,
        origin_um=(10.0, 20.0, 30.0),
        space="zn13x_roi",
        label_names={0: "pore", 2: "na_13x", 7: "zn_13x"},
    )
    vset = VolumeSet(img_vol, labels={"phase": lbl_vol})

    store_path = write_ome_zarr(vset, tmp_path / "triphase.ome.zarr", multiscale_levels=3)

    # Verify NGFF v0.4 JSON ordering: "scale" MUST appear before "translation"
    root_zattrs = json.loads((store_path / ".zattrs").read_text(encoding="utf-8"))
    ds_lvl1 = root_zattrs["multiscales"][0]["datasets"][1]
    tfm_types = [t["type"] for t in ds_lvl1["coordinateTransformations"]]
    assert tfm_types == ["scale", "translation"]
    assert np.allclose(ds_lvl1["coordinateTransformations"][0]["scale"], [1.6, 1.6, 1.6])
    # Level 1 index-0 cell centre shifted by +0.5 * 0.8 = +0.4 um
    assert np.allclose(ds_lvl1["coordinateTransformations"][1]["translation"], [10.4, 20.4, 30.4])

    # Provenance stamped automatically
    prov = root_zattrs["basecoat_provenance"]
    assert prov["basecoat_version"] == __version__
    assert "written_at_utc" in prov

    # Round-trip level 0 and level 1 as VolumeSet
    rt_lvl0 = read_ome_zarr(store_path, level=0, as_set=True)
    assert np.allclose(rt_lvl0.image.data, img_arr)
    assert rt_lvl0.labels["phase"].label_names == {0: "pore", 2: "na_13x", 7: "zn_13x"}

    rt_lvl1 = read_ome_zarr(store_path, level=1, as_set=True)
    lvl1_lbl = rt_lvl1.labels["phase"]
    assert lvl1_lbl.shape == (8, 8, 8)
    assert set(np.unique(lvl1_lbl.data)).issubset({0, 2, 7})
    assert lvl1_lbl.data[0, 0, 0] == 7
    assert lvl1_lbl.data[0, 0, 1] == 2
    assert lvl1_lbl.data[1, 1, 1] == 2


def test_ome_zarr_region_read_never_touches_distant_chunks(tmp_path: Path) -> None:
    arr = np.arange(64 * 64 * 64, dtype=np.int32).reshape(64, 64, 64)
    vol = Volume.from_array(arr, voxel_size_um=1.0)
    store_path = write_ome_zarr(
        vol, tmp_path / "chunked.ome.zarr", multiscale_levels=1, chunks=(16, 16, 16)
    )

    # Corrupt a distant chunk (chunk (3, 3, 3) covering [48:64, 48:64, 48:64]) on disk
    distant_chunk = store_path / "0" / "3" / "3" / "3"
    assert distant_chunk.is_file()
    distant_chunk.write_bytes(b"corrupted_bytes_not_valid_blosc")

    # Reading chunk (0, 0, 0) via open_volume().read_region() succeeds without touching chunk (3, 3, 3)
    with open_volume(store_path) as lv:
        sub = lv.read_region((slice(0, 16), slice(0, 16), slice(0, 16)))
        assert np.array_equal(sub.data, arr[0:16, 0:16, 0:16])
