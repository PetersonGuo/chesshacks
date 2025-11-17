import c_helpers
import pytest


@pytest.fixture
def make_state():
    """Return a helper that constructs BitboardState objects from FEN strings."""

    def _make(fen: str) -> c_helpers.BitboardState:
        return c_helpers.BitboardState(fen)

    return _make
