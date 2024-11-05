from __future__ import annotations

import importlib.metadata

import pyfeather as m


def test_version():
    assert importlib.metadata.version("pyfeather") == m.__version__
