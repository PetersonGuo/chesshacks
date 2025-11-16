#!/usr/bin/env python3
"""Pytest tests for razoring, futility pruning, and SEE features."""

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


@pytest.fixture
def losing_fen():
    """Position where White is down a knight."""
    return "rnbqkb1r/pppppppp/8/8/8/8/1PPPPPPP/RNBQKB1R w KQkq - 0 1"


def test_basic_search_with_pruning(starting_fen):
    """Test basic search with all pruning features."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    best_move = c_helpers.get_best_move_uci(
        starting_fen, 3, c_helpers.evaluate_with_pst, tt, 0, killers, history
    )

    assert isinstance(best_move, str)
    assert len(best_move) >= 4  # UCI format
    assert len(tt) > 0


def test_search_at_multiple_depths(starting_fen):
    """Test pruning features at various depths."""
    tt = c_helpers.TranspositionTable()

    for depth in [2, 3, 4, 5]:
        tt.clear()

        best_move = c_helpers.get_best_move_uci(
            starting_fen, depth, c_helpers.evaluate_with_pst, tt
        )

        assert isinstance(best_move, str)
        assert len(best_move) >= 4


def test_tactical_position_with_see(tactical_fen):
    """Test SEE on tactical position with captures."""
    tt = c_helpers.TranspositionTable()

    import time

    start = time.time()
    best_move = c_helpers.get_best_move_uci(
        tactical_fen, 5, c_helpers.evaluate_with_pst, tt
    )
    elapsed = time.time() - start

    assert isinstance(best_move, str)
    assert len(tt) > 0
    assert elapsed < 10.0  # Should complete reasonably fast


def test_razoring_on_losing_position(losing_fen):
    """Test razoring speeds up clearly bad positions."""
    tt = c_helpers.TranspositionTable()

    score = c_helpers.alpha_beta_optimized(
        losing_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
    )

    assert isinstance(score, int)
    assert score < 0  # Position is losing for White
    assert len(tt) > 0


def test_futility_pruning_shallow_depth(starting_fen):
    """Test futility pruning at shallow depths."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        starting_fen,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        1,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0


def test_see_move_ordering(tactical_fen):
    """Test SEE improves move ordering in tactical positions."""
    tt = c_helpers.TranspositionTable()

    best_move = c_helpers.get_best_move_uci(
        tactical_fen, 4, c_helpers.evaluate_with_pst, tt
    )

    assert isinstance(best_move, str)
    assert len(best_move) >= 4


def test_pruning_with_parallel_search(starting_fen):
    """Test pruning features work with parallel search."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        starting_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        4,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0


def test_razoring_futility_integration(tactical_fen):
    """Test razoring and futility work together."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    import time

    start = time.time()
    score = c_helpers.alpha_beta_optimized(
        tactical_fen,
        5,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        1,
        killers,
        history,
    )
    elapsed = time.time() - start

    assert isinstance(score, int)
    assert len(tt) > 0
    # Pruning should keep search fast
    assert elapsed < 15.0


def test_get_best_move_with_all_tables(starting_fen):
    """Test get_best_move_uci with all data structures."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    best_move = c_helpers.get_best_move_uci(
        starting_fen, 4, c_helpers.evaluate_with_pst, tt, 0, killers, history
    )

    assert isinstance(best_move, str)
    assert len(best_move) >= 4
    assert len(tt) > 0
