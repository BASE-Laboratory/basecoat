# basecoat

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**`basecoat`** is the Layer 1 core plumbing library for the [BASE Laboratory](https://github.com/BASE-Laboratory) image-based modelling ecosystem (`ECOSYSTEM.md`). It provides a single-array `Volume` container, co-registered `VolumeSet`, non-materialising `LazyVolume` region I/O (OME-Zarr v0.4, HDF5, TIFF, `.npy`, and raw binary), pullback 3-D coordinate transforms (`AffineTransform3D`, `SimilarityTransform3D`), cell-centred resampling, and SHA-256 dataset fetching—with lightweight scientific dependencies (`numpy`, `scipy`, `h5py`, `tifffile`, `zarr<3`, `numcodecs`).

**Maintainer:** James Le Houx (`james.lehoux@gre.ac.uk`)

## Installation

```bash
pip install basecoat
```

Optional extras:

```bash
pip install "basecoat[sitk]"    # SimpleITK transform bridge (.to_sitk() / .from_sitk())
pip install "basecoat[dev]"     # pytest, ruff, mypy, SimpleITK, ome-zarr validator
```

## Quick example

```python
import numpy as np
import basecoat as bc

raw = np.random.default_rng(0).normal(size=(64, 64, 64)).astype(np.float32)
labels = (raw > 0.25).astype(np.int32)

img = bc.Volume.from_array(raw, voxel_size_um=0.8125, space="i13_4x")
seg = bc.Volume.from_array(
    labels,
    voxel_size_um=0.8125,
    space="i13_4x",
    label_names={0: "pore", 1: "solid"},
)

# Convert physical spacing explicitly at solver boundaries (metres)
dz_m, dy_m, dx_m = img.spacing_in("m")

# Write co-registered OME-Zarr v0.4 multiscale store (mode-downsampled label pyramid)
vset = bc.VolumeSet(img, labels={"phase": seg})
bc.write_ome_zarr(vset, "sample.ome.zarr", overwrite=True)

# Lazy region read (never loads the full volume into RAM)
with bc.open_volume("sample.ome.zarr") as lv:
    roi = lv.read_region((slice(16, 48), slice(16, 48), slice(16, 48)))
```

## Coordinate and transform conventions

1. **Index-0 cell-centre coordinates:** `AxisSpec.offset` is the physical coordinate of the **centre** of voxel `0`, so voxel `i` has its centre at `offset + i * scale`. Coarsening a spatial axis by factor `k` via `coarsen_axes` shifts the coarse level's index-`0` centre to `offset + 0.5 * (k - 1) * scale`, preserving outer voxel faces.
2. **Pullback transforms (`fixed -> moving`):** `AffineTransform3D` and `SimilarityTransform3D` map physical points from `from_space` (the target/output grid) to `to_space` (the moving/input grid) in `('z', 'y', 'x')` order, matching `scipy.ndimage.affine_transform` and `SimpleITK.ResampleImageFilter`.
3. **Zero silent unit guessing:** Uncalibrated volumes have `unit="voxel"`; calling `vol.spacing_in("m")` raises `UnitConversionError` explaining how to attach spacing.

## Ecosystem architecture

See [`ECOSYSTEM.md`](ECOSYSTEM.md) for the canonical four-layer dependency rules, data contract, two-consumer admission rule, and repository registry across BASE Laboratory.

## Licence

BSD 3-Clause. See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).
