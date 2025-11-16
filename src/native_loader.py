"""Helpers for ensuring the compiled c_helpers module is available.

This attempts to import c_helpers, and if the module is missing it will trigger
the CMake build (via build.sh) automatically. The logic lives in a dedicated
module so every entrypoint can rely on the same behavior.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
BUILD_SCRIPT = PROJECT_ROOT / "build.sh"


def _ensure_sys_path() -> None:
    """Ensure the build directory and project root are importable."""
    for path in (BUILD_DIR, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def ensure_c_helpers():
    """Import the c_helpers extension, building it on demand if required."""
    _ensure_sys_path()
    try:
        return importlib.import_module("c_helpers")
    except ModuleNotFoundError as exc:
        if not BUILD_SCRIPT.exists():
            raise RuntimeError(
                f"c_helpers module not found and build script missing at {BUILD_SCRIPT}"
            ) from exc

        print("=" * 80)
        print("c_helpers module not found - running build.sh automatically...")
        print("=" * 80)

        # Run the build script and stream its output so the user can diagnose failures.
        subprocess.run(
            ["/bin/bash", str(BUILD_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

        # After a successful build, try importing again.
        _ensure_sys_path()
        return importlib.import_module("c_helpers")


__all__ = ["ensure_c_helpers"]
