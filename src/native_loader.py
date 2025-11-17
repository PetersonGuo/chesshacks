"""Helpers for ensuring the compiled c_helpers module is available.

This attempts to import c_helpers, and if the module is missing it will trigger
the CMake build (via build.sh) automatically. The logic lives in a dedicated
module so every entrypoint can rely on the same behavior.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"


def _ensure_sys_path() -> None:
    """Ensure the build directory and project root are importable."""
    for path in (BUILD_DIR, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def ensure_c_helpers():
    """Import the c_helpers extension."""
    _ensure_sys_path()
    return importlib.import_module("c_helpers")


__all__ = ["ensure_c_helpers"]
