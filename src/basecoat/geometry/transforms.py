"""Pullback 3-D coordinate transforms with explicit space, unit, and axis-order contracts.

Convention
----------
Transforms follow the **pullback** (resampling) direction used by SimpleITK and
``scipy.ndimage.affine_transform``: a transform maps points in ``from_space``
(the **fixed / output** grid) to ``to_space`` (the **moving / input** grid) in
``('z', 'y', 'x')`` coordinate order by default.

When interfacing with SimpleITK via :meth:`to_sitk` / :meth:`from_sitk`, the
``('z', 'y', 'x') <-> ('x', 'y', 'z')`` axis permutation and SimpleITK's
scale-included ``Similarity3DTransform.GetMatrix()`` ($M = s R$) convention are
handled in one place.

Adapted and generalised from ``MultimodalBenchmark/registration/i13_ladder.py``
(James Le Houx, Calum Green).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import SpaceMismatchError
from ..volume import canonicalize_unit

_PERM_ZYX_XYZ = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _itk_euler_xyz_matrix(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray[Any, Any]:
    """Exact 3x3 XYZ matrix matching ``SimpleITK.Euler3DTransform`` (``ComputeZYX=False``: ``Rz @ Rx @ Ry``)."""
    cx, sx = math.cos(rx_rad), math.sin(rx_rad)
    cy, sy = math.cos(ry_rad), math.sin(ry_rad)
    cz, sz = math.cos(rz_rad), math.sin(rz_rad)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ rx @ ry


def _require_sitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for .to_sitk() / .from_sitk(); install via `pip install basecoat[sitk]`."
        ) from exc
    return sitk


@dataclass(frozen=True)
class AffineTransform3D:
    """Homogeneous 4x4 3-D affine transform mapping ``from_space`` -> ``to_space``."""

    matrix_4x4: np.ndarray[Any, Any]
    from_space: str
    to_space: str
    unit: str = "um"
    axis_order: tuple[str, str, str] = ("z", "y", "x")

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix_4x4, dtype=np.float64)
        if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
            raise ValueError(
                f"AffineTransform3D requires a finite (4, 4) matrix, got shape={mat.shape}."
            )
        if not np.allclose(mat[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError(
                f"Bottom row of homogeneous 4x4 matrix must be [0, 0, 0, 1], got {mat[3, :]!r}."
            )
        det = float(np.linalg.det(mat[:3, :3]))
        if abs(det) < 1e-12:
            raise ValueError(
                f"AffineTransform3D linear 3x3 block must be invertible, got det={det:.3e}."
            )
        src = self.from_space.strip()
        dst = self.to_space.strip()
        if not src or not dst:
            raise ValueError(
                "AffineTransform3D requires non-empty from_space and to_space identifiers."
            )
        if len(self.axis_order) != 3 or len(set(self.axis_order)) != 3:
            raise ValueError(
                f"axis_order must contain 3 distinct axis names, got {self.axis_order!r}."
            )

        mat_ro = mat.copy()
        mat_ro.setflags(write=False)
        object.__setattr__(self, "matrix_4x4", mat_ro)
        object.__setattr__(self, "from_space", src)
        object.__setattr__(self, "to_space", dst)
        object.__setattr__(self, "unit", canonicalize_unit(self.unit))
        object.__setattr__(
            self,
            "axis_order",
            (str(self.axis_order[0]), str(self.axis_order[1]), str(self.axis_order[2])),
        )

    @classmethod
    def identity(
        cls,
        *,
        space: str = "world",
        unit: str = "um",
        axis_order: tuple[str, str, str] = ("z", "y", "x"),
    ) -> AffineTransform3D:
        return cls(
            np.eye(4, dtype=np.float64),
            from_space=space,
            to_space=space,
            unit=unit,
            axis_order=axis_order,
        )

    @classmethod
    def from_linear_and_offset(
        cls,
        linear_3x3: np.ndarray[Any, Any],
        offset_3: Sequence[float] | np.ndarray[Any, Any],
        *,
        from_space: str,
        to_space: str,
        unit: str = "um",
        axis_order: tuple[str, str, str] = ("z", "y", "x"),
    ) -> AffineTransform3D:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = np.asarray(linear_3x3, dtype=np.float64)
        mat[:3, 3] = np.asarray(offset_3, dtype=np.float64)
        return cls(mat, from_space=from_space, to_space=to_space, unit=unit, axis_order=axis_order)

    @property
    def linear_matrix(self) -> np.ndarray[Any, Any]:
        return self.matrix_4x4[:3, :3]

    @property
    def translation_vector(self) -> np.ndarray[Any, Any]:
        return self.matrix_4x4[:3, 3]

    def transform_points(
        self, points: np.ndarray[Any, Any] | Sequence[Sequence[float]]
    ) -> np.ndarray[Any, Any]:
        """Map ``(..., 3)`` coordinates in ``from_space`` (ordered as ``axis_order``) into ``to_space``."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError(f"Expected points with trailing dimension 3, got shape={pts.shape}.")
        mapped: np.ndarray[Any, Any] = pts @ self.linear_matrix.T + self.translation_vector
        return mapped

    def inverse(self) -> AffineTransform3D:
        """Return exact inverse transform mapping ``to_space`` -> ``from_space``."""
        inv_mat = np.linalg.inv(self.matrix_4x4)
        inv_mat[3, :] = [0.0, 0.0, 0.0, 1.0]
        return AffineTransform3D(
            inv_mat,
            from_space=self.to_space,
            to_space=self.from_space,
            unit=self.unit,
            axis_order=self.axis_order,
        )

    def then(self, next_transform: AffineTransform3D) -> AffineTransform3D:
        """Chain ``self`` (``A -> B``) followed by ``next_transform`` (``B -> C``) to produce ``A -> C``."""
        if self.to_space != next_transform.from_space:
            raise SpaceMismatchError(
                f"Cannot chain transform '{self.from_space} -> {self.to_space}' into "
                f"'{next_transform.from_space} -> {next_transform.to_space}': intermediate spaces differ."
            )
        if self.unit != next_transform.unit:
            raise SpaceMismatchError(
                f"Cannot chain transforms with different units ('{self.unit}' vs '{next_transform.unit}')."
            )
        if self.axis_order != next_transform.axis_order:
            raise SpaceMismatchError(
                f"Cannot chain transforms with different axis_order ({self.axis_order} vs {next_transform.axis_order})."
            )
        combined = next_transform.matrix_4x4 @ self.matrix_4x4
        combined[3, :] = [0.0, 0.0, 0.0, 1.0]
        return AffineTransform3D(
            combined,
            from_space=self.from_space,
            to_space=next_transform.to_space,
            unit=self.unit,
            axis_order=self.axis_order,
        )

    def to_sitk(self) -> Any:
        """Convert a ``('z', 'y', 'x')`` transform to a ``SimpleITK.AffineTransform`` in ``(x, y, z)`` order."""
        if self.axis_order != ("z", "y", "x"):
            raise ValueError(
                f"to_sitk() requires axis_order=('z', 'y', 'x'), got {self.axis_order}."
            )
        sitk = _require_sitk()
        a_xyz = _PERM_ZYX_XYZ @ self.linear_matrix @ _PERM_ZYX_XYZ
        t_xyz = _PERM_ZYX_XYZ @ self.translation_vector
        tx = sitk.AffineTransform(3)
        tx.SetMatrix(a_xyz.ravel().tolist())
        tx.SetTranslation(t_xyz.tolist())
        return tx

    @classmethod
    def from_sitk(
        cls,
        sitk_transform: Any,
        *,
        from_space: str,
        to_space: str,
        unit: str = "um",
    ) -> AffineTransform3D:
        """Import a 3-D linear ``SimpleITK`` transform into ``('z', 'y', 'x')`` coordinates."""
        sitk = _require_sitk()
        tx = sitk_transform
        if isinstance(tx, sitk.CompositeTransform):
            if tx.GetNumberOfTransforms() != 1:
                raise ValueError(
                    "CompositeTransform with multiple stages should be converted stage-by-stage."
                )
            tx = tx.GetNthTransform(0)
        mat_xyz = np.asarray(tx.GetMatrix(), dtype=np.float64).reshape(3, 3)
        cen_xyz = np.asarray(tx.GetCenter(), dtype=np.float64)
        trn_xyz = np.asarray(tx.GetTranslation(), dtype=np.float64)
        offset_xyz = trn_xyz + cen_xyz - mat_xyz @ cen_xyz

        a_zyx = _PERM_ZYX_XYZ @ mat_xyz @ _PERM_ZYX_XYZ
        b_zyx = _PERM_ZYX_XYZ @ offset_xyz
        return cls.from_linear_and_offset(
            a_zyx,
            b_zyx,
            from_space=from_space,
            to_space=to_space,
            unit=unit,
            axis_order=("z", "y", "x"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "basecoat.affine_transform_3d.v1",
            "direction": "pullback_fixed_to_moving",
            "from_space": self.from_space,
            "to_space": self.to_space,
            "unit": self.unit,
            "axis_order": list(self.axis_order),
            "matrix_4x4": self.matrix_4x4.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AffineTransform3D:
        raw_axes = payload["axis_order"]
        return cls(
            np.asarray(payload["matrix_4x4"], dtype=np.float64),
            from_space=str(payload["from_space"]),
            to_space=str(payload["to_space"]),
            unit=str(payload["unit"]),
            axis_order=(str(raw_axes[0]), str(raw_axes[1]), str(raw_axes[2])),
        )

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: str | Path) -> AffineTransform3D:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass(frozen=True)
class SimilarityTransform3D:
    """7-DOF 3-D similarity transform in ``('z', 'y', 'x')`` coordinates:

        p_to = scale * rotation_zyx @ (p_from - center_zyx) + center_zyx + translation_zyx

    Note: ``rotation_zyx`` is strictly an unscaled proper orthonormal rotation matrix
    ($R^T R = I$, $\\det R = +1$). When importing from ``SimpleITK.Similarity3DTransform``
    (whose ``GetMatrix()`` returns the already-scaled $s R$), :meth:`from_sitk` factors
    out ``scale`` so it is never applied twice.
    """

    rotation_zyx: np.ndarray[Any, Any]
    scale: float
    center_zyx: tuple[float, float, float]
    translation_zyx: tuple[float, float, float]
    from_space: str
    to_space: str
    unit: str = "um"

    def __post_init__(self) -> None:
        rot = np.asarray(self.rotation_zyx, dtype=np.float64)
        if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
            raise ValueError(f"rotation_zyx must be a finite (3, 3) matrix, got shape={rot.shape}.")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5):
            raise ValueError(
                "SimilarityTransform3D.rotation_zyx must be orthonormal (unscaled R^T R = I); "
                "if passing SimpleITK's scaled GetMatrix(), divide out scale first or use .from_sitk()."
            )
        det = float(np.linalg.det(rot))
        if det < 0.0:
            raise ValueError(f"rotation_zyx must be a proper rotation (det=+1), got det={det:.6f}.")
        sc = float(self.scale)
        if not math.isfinite(sc) or sc <= 0.0:
            raise ValueError(
                f"SimilarityTransform3D.scale must be finite and > 0, got {self.scale!r}."
            )
        cen = (float(self.center_zyx[0]), float(self.center_zyx[1]), float(self.center_zyx[2]))
        trn = (
            float(self.translation_zyx[0]),
            float(self.translation_zyx[1]),
            float(self.translation_zyx[2]),
        )
        if not (all(math.isfinite(v) for v in cen) and all(math.isfinite(v) for v in trn)):
            raise ValueError("center_zyx and translation_zyx must be finite.")
        src = self.from_space.strip()
        dst = self.to_space.strip()
        if not src or not dst:
            raise ValueError("from_space and to_space must be non-empty strings.")

        rot_ro = rot.copy()
        rot_ro.setflags(write=False)
        object.__setattr__(self, "rotation_zyx", rot_ro)
        object.__setattr__(self, "scale", sc)
        object.__setattr__(self, "center_zyx", cen)
        object.__setattr__(self, "translation_zyx", trn)
        object.__setattr__(self, "from_space", src)
        object.__setattr__(self, "to_space", dst)
        object.__setattr__(self, "unit", canonicalize_unit(self.unit))

    @classmethod
    def from_euler_xyz(
        cls,
        *,
        euler_xyz_rad: Sequence[float],
        scale: float,
        translation_xyz: Sequence[float],
        center_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        from_space: str,
        to_space: str,
        unit: str = "um",
    ) -> SimilarityTransform3D:
        """Construct from ITK/SimpleITK XYZ Euler angles ``(rx, ry, rz)`` without requiring SimpleITK."""
        if len(euler_xyz_rad) != 3 or len(translation_xyz) != 3 or len(center_xyz) != 3:
            raise ValueError(
                "euler_xyz_rad, translation_xyz, and center_xyz must each have 3 elements."
            )
        r_xyz = _itk_euler_xyz_matrix(
            float(euler_xyz_rad[0]),
            float(euler_xyz_rad[1]),
            float(euler_xyz_rad[2]),
        )
        r_zyx = _PERM_ZYX_XYZ @ r_xyz @ _PERM_ZYX_XYZ
        cen_zyx = (float(center_xyz[2]), float(center_xyz[1]), float(center_xyz[0]))
        trn_zyx = (float(translation_xyz[2]), float(translation_xyz[1]), float(translation_xyz[0]))
        return cls(
            rotation_zyx=r_zyx,
            scale=float(scale),
            center_zyx=cen_zyx,
            translation_zyx=trn_zyx,
            from_space=from_space,
            to_space=to_space,
            unit=unit,
        )

    def to_affine(self) -> AffineTransform3D:
        """Convert to an exact :class:`AffineTransform3D` in ``('z', 'y', 'x')`` order."""
        lin = self.scale * self.rotation_zyx
        cen = np.asarray(self.center_zyx, dtype=np.float64)
        trn = np.asarray(self.translation_zyx, dtype=np.float64)
        offset = trn + cen - lin @ cen
        return AffineTransform3D.from_linear_and_offset(
            lin,
            offset,
            from_space=self.from_space,
            to_space=self.to_space,
            unit=self.unit,
            axis_order=("z", "y", "x"),
        )

    def transform_points(
        self, points_zyx: np.ndarray[Any, Any] | Sequence[Sequence[float]]
    ) -> np.ndarray[Any, Any]:
        """Map ``(..., 3)`` points in ``('z', 'y', 'x')`` from ``from_space`` to ``to_space``."""
        return self.to_affine().transform_points(points_zyx)

    def inverse(self) -> SimilarityTransform3D:
        """Return exact 7-DOF inverse mapping ``to_space`` -> ``from_space``."""
        inv_scale = 1.0 / self.scale
        inv_rot = self.rotation_zyx.T
        trn = np.asarray(self.translation_zyx, dtype=np.float64)
        # p_to = s R (p_from - c) + c + t  =>  p_from = (1/s) R^T (p_to - c) + c - (1/s) R^T t
        inv_trn = -(inv_scale * (inv_rot @ trn))
        return SimilarityTransform3D(
            rotation_zyx=inv_rot,
            scale=inv_scale,
            center_zyx=self.center_zyx,
            translation_zyx=(float(inv_trn[0]), float(inv_trn[1]), float(inv_trn[2])),
            from_space=self.to_space,
            to_space=self.from_space,
            unit=self.unit,
        )

    def to_sitk(self) -> Any:
        """Convert to ``SimpleITK.Similarity3DTransform`` in ``(x, y, z)`` order."""
        sitk = _require_sitk()
        r_xyz = _PERM_ZYX_XYZ @ self.rotation_zyx @ _PERM_ZYX_XYZ
        cen_xyz = [self.center_zyx[2], self.center_zyx[1], self.center_zyx[0]]
        trn_xyz = [self.translation_zyx[2], self.translation_zyx[1], self.translation_zyx[0]]
        tx = sitk.Similarity3DTransform()
        tx.SetCenter(cen_xyz)
        tx.SetMatrix(r_xyz.ravel().tolist())
        tx.SetScale(float(self.scale))
        tx.SetTranslation(trn_xyz)
        return tx

    @classmethod
    def from_sitk(
        cls,
        sitk_transform: Any,
        *,
        from_space: str,
        to_space: str,
        unit: str = "um",
    ) -> SimilarityTransform3D:
        """Import a ``SimpleITK.Similarity3DTransform``, factoring ``scale`` out of ``GetMatrix()``."""
        sitk = _require_sitk()
        tx = sitk_transform
        if isinstance(tx, sitk.CompositeTransform):
            tx = tx.GetNthTransform(tx.GetNumberOfTransforms() - 1)
        sim = sitk.Similarity3DTransform(tx)
        scale = float(sim.GetScale())
        scaled_mat_xyz = np.asarray(sim.GetMatrix(), dtype=np.float64).reshape(3, 3)
        r_xyz = scaled_mat_xyz / scale
        r_zyx = _PERM_ZYX_XYZ @ r_xyz @ _PERM_ZYX_XYZ
        cen_xyz = sim.GetCenter()
        trn_xyz = sim.GetTranslation()
        return cls(
            rotation_zyx=r_zyx,
            scale=scale,
            center_zyx=(float(cen_xyz[2]), float(cen_xyz[1]), float(cen_xyz[0])),
            translation_zyx=(float(trn_xyz[2]), float(trn_xyz[1]), float(trn_xyz[0])),
            from_space=from_space,
            to_space=to_space,
            unit=unit,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "basecoat.similarity_transform_3d.v1",
            "direction": "pullback_fixed_to_moving",
            "from_space": self.from_space,
            "to_space": self.to_space,
            "unit": self.unit,
            "axis_order": ["z", "y", "x"],
            "rotation_zyx": self.rotation_zyx.tolist(),
            "scale": self.scale,
            "center_zyx": list(self.center_zyx),
            "translation_zyx": list(self.translation_zyx),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SimilarityTransform3D:
        cen = payload["center_zyx"]
        trn = payload["translation_zyx"]
        return cls(
            rotation_zyx=np.asarray(payload["rotation_zyx"], dtype=np.float64),
            scale=float(payload["scale"]),
            center_zyx=(float(cen[0]), float(cen[1]), float(cen[2])),
            translation_zyx=(float(trn[0]), float(trn[1]), float(trn[2])),
            from_space=str(payload["from_space"]),
            to_space=str(payload["to_space"]),
            unit=str(payload["unit"]),
        )

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: str | Path) -> SimilarityTransform3D:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
