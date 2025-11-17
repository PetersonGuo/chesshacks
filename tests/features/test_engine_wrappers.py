"""
Smoke tests for the Python-side engine helpers in `src/engine.py`.

These wrappers convert FEN strings into `BitboardState` objects before
invoking the native bindings. The tests below make sure the helpers still
produce usable scores and stay in sync with the lower-level APIs.
"""

from __future__ import annotations

import c_helpers
import pytest

from src import engine

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_engine_alpha_beta_round_trip():
    """engine.alpha_beta should agree with search_position for the same inputs."""
    depth = 3
    score_direct = engine.alpha_beta(STARTING_FEN, depth)
    score_search = engine.search_position(STARTING_FEN, depth, num_threads=1)

    assert isinstance(score_direct, int)
    assert isinstance(score_search, int)
    assert score_direct == score_search


def test_engine_bitboard_conversion_matches_native(make_state):
    """bitboard_from_fen should mirror the direct BitboardState constructor."""
    state_from_engine = engine.bitboard_from_fen(STARTING_FEN)
    state_direct = make_state(STARTING_FEN)

    assert isinstance(state_from_engine, c_helpers.BitboardState)
    assert state_from_engine.to_fen() == state_direct.to_fen()


@pytest.mark.skipif(
    not hasattr(c_helpers, "alpha_beta_cuda"), reason="CUDA bindings disabled"
)
def test_engine_cuda_wrapper_matches_cpu():
    """alpha_beta_cuda should fall back to the CPU path when CUDA is absent."""
    depth = 2
    cpu_score = engine.alpha_beta(STARTING_FEN, depth)
    cuda_score = engine.alpha_beta_cuda(STARTING_FEN, depth)

    assert isinstance(cuda_score, int)
    assert abs(cuda_score - cpu_score) <= 400

