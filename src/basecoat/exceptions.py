"""Custom exceptions for basecoat with actionable remediation messages."""

from __future__ import annotations


class BasecoatError(Exception):
    """Base class for all basecoat domain errors."""


class UnitConversionError(BasecoatError):
    """Raised when converting uncalibrated ('voxel') or incompatible axis units."""


class SpaceMismatchError(BasecoatError):
    """Raised when composing transforms or pairing volumes across mismatched spaces or grids."""


class ChecksumError(BasecoatError):
    """Raised when a fetched file's SHA-256 digest does not match the expected value."""
