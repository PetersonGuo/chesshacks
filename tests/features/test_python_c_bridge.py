"""
Python ↔ C++ bridge tests that exercise the nanobind module directly.

These tests ensure the exposed helpers accept BitboardState objects created on
the Python side and that their results stay in sync with the native pipeline.
"""

from __future__ import annotations

import c_helpers
import chess

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_batch_evaluate_matches_python(make_state):
    states = [
        make_state(STARTING_FEN),
        make_state("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3"),
        make_state(
            "rnbq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"
        ),
    ]
    sequential = [c_helpers.evaluate(state) for state in states]
    batch = c_helpers.batch_evaluate_mt(states, num_threads=0)
    assert batch == sequential


def test_pgn_to_bitboard_round_trip():
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"
    bitboard_state = c_helpers.pgn_to_bitboard(pgn)
    reference = chess.Board()
    reference.push_san("e4")
    reference.push_san("e5")
    reference.push_san("Nf3")
    reference.push_san("Nc6")
    reference.push_san("Bb5")
    reference.push_san("a6")
    assert bitboard_state.to_fen() == reference.fen()


def test_find_best_move_builtin_returns_state(make_state):
    state = make_state(STARTING_FEN)
    result = c_helpers.find_best_move_builtin(state, depth=2)
    assert isinstance(result, c_helpers.BitboardState)
    assert result.to_fen() != state.to_fen()
