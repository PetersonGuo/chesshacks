#!/usr/bin/env python3
"""Pytest tests for Opening Book, Multi-PV, and Tablebase features."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.fixture
def starting_fen():
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def tactical_fen():
    return "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def test_opening_book_creation():
    """Test opening book can be created."""
    book = c_helpers.OpeningBook()
    assert book is not None
    assert hasattr(book, "is_loaded")
    assert hasattr(book, "load")
    assert hasattr(book, "probe")


def test_opening_book_not_loaded_by_default():
    """Test opening book is not loaded by default."""
    book = c_helpers.OpeningBook()
    assert book.is_loaded() is False


def test_opening_book_load_nonexistent():
    """Test loading nonexistent book file fails gracefully."""
    book = c_helpers.OpeningBook()
    result = book.load("nonexistent_book.bin")
    assert result is False
    assert book.is_loaded() is False


def test_opening_book_probe_methods_exist():
    """Test opening book has all probe methods."""
    book = c_helpers.OpeningBook()
    assert hasattr(book, "probe")
    assert hasattr(book, "probe_best")
    assert hasattr(book, "probe_weighted")


def test_multi_pv_search_basic(starting_fen):
    """Test Multi-PV search on starting position."""
    pv_lines = c_helpers.multi_pv_search(
        starting_fen, depth=4, num_lines=3, evaluate=c_helpers.evaluate_with_pst
    )

    assert len(pv_lines) > 0
    assert len(pv_lines) <= 3  # Requested 3 lines

    # Each line should have required attributes
    for line in pv_lines:
        assert hasattr(line, "uci_move")
        assert hasattr(line, "score")
        assert hasattr(line, "depth")
        assert hasattr(line, "pv")


def test_multi_pv_search_tactical(tactical_fen):
    """Test Multi-PV on tactical position."""
    pv_lines = c_helpers.multi_pv_search(
        tactical_fen, depth=5, num_lines=5, evaluate=c_helpers.evaluate_with_pst
    )

    assert len(pv_lines) > 0
    assert len(pv_lines) <= 5

    # Scores should be ordered (best first)
    for i in range(len(pv_lines) - 1):
        # Allow some tolerance due to search variations
        assert pv_lines[i].score >= pv_lines[i + 1].score - 100


def test_multi_pv_single_line(starting_fen):
    """Test Multi-PV with single line (should work like normal search)."""
    pv_lines = c_helpers.multi_pv_search(
        starting_fen, depth=3, num_lines=1, evaluate=c_helpers.evaluate_with_pst
    )

    assert len(pv_lines) == 1
    assert pv_lines[0].depth == 3


def test_multi_pv_deeper_search(starting_fen):
    """Test Multi-PV at deeper depth."""
    pv_lines = c_helpers.multi_pv_search(
        starting_fen, depth=4, num_lines=3, evaluate=c_helpers.evaluate_with_pst
    )

    assert len(pv_lines) > 0

    for line in pv_lines:
        assert line.depth == 4
        assert isinstance(line.score, int)
        assert isinstance(line.uci_move, str)


def test_multi_pv_move_format(starting_fen):
    """Test Multi-PV returns valid UCI move format."""
    pv_lines = c_helpers.multi_pv_search(
        starting_fen, depth=3, num_lines=3, evaluate=c_helpers.evaluate_with_pst
    )

    assert len(pv_lines) > 0

    for line in pv_lines:
        # UCI moves should be 4-5 characters (e.g., "e2e4" or "e7e8q")
        assert len(line.uci_move) >= 4
        assert len(line.uci_move) <= 5


def test_multi_pv_with_simple_evaluate(starting_fen):
    """Test Multi-PV with custom evaluation function."""

    def simple_eval(fen: str) -> int:
        piece_values = {
            "P": 100,
            "N": 320,
            "B": 330,
            "R": 500,
            "Q": 900,
            "K": 0,
            "p": -100,
            "n": -320,
            "b": -330,
            "r": -500,
            "q": -900,
            "k": 0,
        }
        return sum(piece_values.get(c, 0) for c in fen.split()[0])

    pv_lines = c_helpers.multi_pv_search(
        starting_fen, depth=3, num_lines=2, evaluate=simple_eval
    )

    assert len(pv_lines) > 0
    assert len(pv_lines) <= 2
