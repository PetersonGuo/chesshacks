"""
Shared benchmark bootstrap.

Pytest automatically executes this file, and benchmark scripts can rely on the
resulting sys.path so they never need to manipulate imports individually.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
SRC_DIR = REPO_ROOT / "src"

for path in (SRC_DIR, BUILD_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

