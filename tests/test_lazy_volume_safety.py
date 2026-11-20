"""Tests ensuring LazyVolume never silently materialises full datasets into RAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from basecoat import Volume, open_volume, write_volume


def test_lazy_volume_blocks_implicit_asarray_and_enforces_max_bytes(tmp_path: Path) -> None:
    arr = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    vol = Volume.from_array(arr, voxel_size_um=1.25, origin_um=(5.0, 10.0, 15.0), space="h5_grid")
    h5_path = write_volume(vol, tmp_path / "cube.h5")

    with open_volume(h5_path) as lv:
        assert lv.shape == (16, 16, 16)
        assert lv.nbytes == arr.nbytes

        # 1. Implicit np.asarray(lv) MUST fail loudly rather than materialising or wrapping object
        with pytest.raises(TypeError, match="read_region"):
            np.asarray(lv)

        # 2. Explicit to_volume() fails before touching disk if nbytes > max_bytes
        with pytest.raises(MemoryError, match="read_region"):
            lv.to_volume(max_bytes=1024)

        # 3. Bounded region read succeeds and shifts cell-centre offsets accurately
        sub = lv.read_region((slice(2, 6), slice(4, 8), slice(1, 5)))
        assert sub.shape == (4, 4, 4)
        assert np.array_equal(sub.data, arr[2:6, 4:8, 1:5])
        assert np.allclose(
            sub.origin_in("um"),
            (5.0 + 2 * 1.25, 10.0 + 4 * 1.25, 15.0 + 1 * 1.25),
        )

    assert lv.closed
    with pytest.raises(ValueError, match="closed"):
        lv.read_region((slice(0, 2), slice(0, 2), slice(0, 2)))
