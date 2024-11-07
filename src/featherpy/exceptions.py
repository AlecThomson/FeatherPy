from __future__ import annotations


class UnitError(Exception):
    """Raised when a unit is not recognized or is not supported."""


class ShapeError(Exception):
    """Raised when a shape is not recognized or is not supported."""
