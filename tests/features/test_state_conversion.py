"""
PGN/FEN/Bitboard conversion regression tests.

These combine the previous PGN-to-FEN matrix and the bitboard regression suite
into a single pytest module to keep coverage focused without many files.
"""

from __future__ import annotations

import io

import c_helpers
import chess.pgn
import pytest

from src import engine


def _state_from_pgn(pgn: str) -> c_helpers.BitboardState:
    """Apply a PGN string to a board and convert to BitboardState."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None, f"Failed to parse PGN: {pgn}"
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return engine._board_to_bitboard_state(board)


@pytest.mark.parametrize(
    ("name", "pgn", "expected"),
    [
        ("Starting Position", "", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"),
        ("Single Move", "1. e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"),
        ("Two Moves", "1. e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"),
        (
            "Italian Game",
            "1. e4 e5 2. Nf3 Nc6 3. Bc4",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        ),
        (
            "Scholar Mate Setup",
            "1. e4 e5 2. Bc4 Nc6 3. Qh5",
            "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR",
        ),
        (
            "Sicilian Defense",
            "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4",
            "rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R",
        ),
        (
            "French Defense",
            "1. e4 e6 2. d4 d5 3. Nc3",
            "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR",
        ),
        (
            "King's Gambit",
            "1. e4 e5 2. f4 exf4",
            "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR",
        ),
    ],
)
def test_pgn_to_fen_conversion(name: str, pgn: str, expected: str):
    result_fen = c_helpers.pgn_to_fen(pgn)
    assert result_fen.split()[0] == expected, f"{name}: mismatch"


@pytest.mark.parametrize(
    "pgn",
    [
        "1. xyz",
        "1. e5",
        "1. Nf3 Nf6 2. Nf3",
    ],
)
def test_invalid_pgn_returns_valid_fen(pgn: str):
    result = c_helpers.pgn_to_fen(pgn)
    assert len(result.split()) == 6  # FEN has 6 fields even if fallback


@pytest.mark.parametrize(
    "pgn",
    [
        "1. e4 Nh6 2. d4 Rg8 3. Nf3 Rh8 4. Ne5 Rg8 5. Bxh6 Rh8 6. Nc3",
        "1. e4 e5 2. Nc3",
    ],
)
def test_bitboard_state_from_pgn_runs_search(pgn: str):
    state = _state_from_pgn(pgn)
    tt = c_helpers.TranspositionTable()
    score = c_helpers.alpha_beta(
        state,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        1,
    )
    assert isinstance(score, int)
