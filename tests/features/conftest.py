from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c_helpers
import pytest


@pytest.fixture
def make_state():
    """Return a helper that constructs BitboardState objects from FEN strings."""

    def _make(fen: str) -> c_helpers.BitboardState:
        return c_helpers.BitboardState(fen)

    return _make
