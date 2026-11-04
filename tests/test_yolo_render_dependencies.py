"""The YOLO chart renderer's imports resolve in a fresh install.

A transitive dependency can drop out of a rebuild (python-dateutil 4 stopped
installing ``six``, which mplfinance imports), and run_yolo_patterns.py then
fails every ticker while exiting 0. Importing the stack here makes CI fail
instead.
"""
from __future__ import annotations

import importlib

import pytest

RENDER_MODULES = ("mplfinance", "matplotlib.pyplot", "PIL.Image")


@pytest.mark.parametrize("module", RENDER_MODULES)
def test_yolo_render_module_imports(module: str) -> None:
    importlib.import_module(module)
