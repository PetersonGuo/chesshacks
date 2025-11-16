#!/usr/bin/env python3
"""
Test PGN to FEN conversion functionality using pytest
Tests various PGN formats and move sequences
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.mark.parametrize(
    "name,pgn,expected",
    [
        (
            "Starting Position (Empty PGN)",
            "",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        ),
        (
            "Single Move (1. e4)",
            "1. e4",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        ),
        (
            "Two Moves (1. e4 e5)",
            "1. e4 e5",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
        ),
        (
            "Italian Game Opening",
            "1. e4 e5 2. Nf3 Nc6 3. Bc4",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        ),
        (
            "Scholar's Mate Setup",
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
        (
            "Castling Kingside",
            "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O",
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1",
        ),
        (
            "Castling Queenside",
            "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. a3 Bxc3+ 6. Qxc3 d6 7. Nf3 Nbd7 8. e3 b6 9. Be2 Bb7 10. O-O",
            "r2q1rk1/pbpn1ppp/1p1ppn2/8/2PP4/P1Q1PN2/1P2BPPP/R1B2RK1",
        ),
    ],
)
def test_pgn_to_fen_conversions(name, pgn, expected):
    """Test PGN to FEN conversion."""
    result_fen = c_helpers.pgn_to_fen(pgn)
    result_pieces = result_fen.split()[0]
    assert (
        result_pieces == expected
    ), f"{name}: Expected {expected}, got {result_pieces}"


@pytest.mark.parametrize(
    "name,pgn",
    [
        ("Invalid move notation", "1. xyz"),
        ("Illegal move", "1. e5"),
        ("Move to occupied square", "1. Nf3 Nf6 2. Nf3"),
    ],
)
def test_invalid_pgn(name, pgn):
    """Test that invalid PGN returns starting position or handles gracefully."""
    result = c_helpers.pgn_to_fen(pgn)
    # Should return a valid FEN string (might be starting position for invalid input)
    assert len(result) > 0, f"{name}: Should return valid FEN"
    assert " " in result, f"{name}: Should have FEN fields separated by spaces"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
