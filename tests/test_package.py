from __future__ import annotations

import importlib.metadata

import featherpy as m


def test_version():
    assert importlib.metadata.version("featherpy") == m.__version__
