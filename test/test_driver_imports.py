"""Every per-example driver module imports cleanly.

This catches wiring breakage after the refactor (wrong ``from config import`` names,
missing ``saa_clt`` symbols, etc.) without running any solve -- the study logic is
guarded behind ``if __name__ == '__main__'`` / ``main()``.  The two examples share the
module names ``model`` / ``config`` / ``clt`` / ..., so each example is imported with
its own directory on ``sys.path`` and the shared names purged in between.
"""

import os
import sys
import importlib

import pytest

from _saa_helpers import EXAMPLES, SCRIPTS

DRIVER_MODULES = ["model", "config", "nominal_saa", "clt", "inference", "coverage"]


@pytest.mark.parametrize("example", EXAMPLES)
def test_driver_modules_import(example):
    example_dir = os.path.join(SCRIPTS, example)
    for name in DRIVER_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, example_dir)
    try:
        for name in DRIVER_MODULES:
            importlib.import_module(name)
    finally:
        if example_dir in sys.path:
            sys.path.remove(example_dir)
        for name in DRIVER_MODULES:
            sys.modules.pop(name, None)
