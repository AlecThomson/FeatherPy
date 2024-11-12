"""
Copyright (c) 2024 Alec Thomson. All rights reserved.

FeatherPy: A python package to combine radio as
"""

from __future__ import annotations

from featherpy.logging import setup_logger

from ._version import version as __version__

logger = setup_logger()

__all__ = ["__version__"]
