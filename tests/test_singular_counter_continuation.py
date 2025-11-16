#!/usr/bin/env python3
"""Pytest tests for singular extensions, counter moves, and continuation history."""

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
    return "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def test_counter_move_table_creation():
    """Test counter move table can be created."""
    counter_moves = c_helpers.CounterMoveTable()
    assert counter_moves is not None
    assert hasattr(counter_moves, "clear")


def test_counter_move_table_clear():
    """Test counter move table clearing."""
    counter_moves = c_helpers.CounterMoveTable()
    counter_moves.clear()
    # Should not raise exception
    assert True


def test_get_best_move_with_counter_moves(starting_fen):
    """Test move retrieval with counter move table."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    best_move = c_helpers.get_best_move_uci(
        starting_fen, 4, c_helpers.evaluate_with_pst, tt, 0, killers, history, counters
    )

    assert isinstance(best_move, str)
    assert len(best_move) >= 4


def test_search_at_various_depths_with_counters(starting_fen):
    """Test counter moves at various depths."""
    counters = c_helpers.CounterMoveTable()

    for depth in [2, 3, 4, 5]:
        move = c_helpers.get_best_move_uci(
            starting_fen,
            depth,
            c_helpers.evaluate_with_pst,
            c_helpers.TranspositionTable(),
            0,
            None,
            None,
            counters,
        )

        assert isinstance(move, str)
        assert len(move) >= 4


def test_tactical_position_with_all_enhancements(tactical_fen):
    """Test tactical position with all enhancements."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    best_move = c_helpers.get_best_move_uci(
        tactical_fen, 5, c_helpers.evaluate_with_pst, tt, 0, killers, history, counters
    )

    assert isinstance(best_move, str)
    assert len(best_move) >= 4


def test_counter_moves_with_transposition_table(starting_fen):
    """Test counter moves integrate with TT."""
    tt = c_helpers.TranspositionTable()
    counters = c_helpers.CounterMoveTable()

    best_move = c_helpers.get_best_move_uci(
        starting_fen, 4, c_helpers.evaluate_with_pst, tt, 0, None, None, counters
    )

    assert isinstance(best_move, str)
    assert len(tt) > 0


def test_all_data_structures_together(starting_fen):
    """Test all data structures work together."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    # First search
    move1 = c_helpers.get_best_move_uci(
        starting_fen, 3, c_helpers.evaluate_with_pst, tt, 1, killers, history, counters
    )

    # Reuse tables for second search
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    move2 = c_helpers.get_best_move_uci(
        after_e4, 3, c_helpers.evaluate_with_pst, tt, 1, killers, history, counters
    )

    assert isinstance(move1, str)
    assert isinstance(move2, str)


def test_counter_moves_with_parallel_search(starting_fen):
    """Test counter moves work with parallel search."""
    tt = c_helpers.TranspositionTable()
    counters = c_helpers.CounterMoveTable()

    best_move = c_helpers.get_best_move_uci(
        starting_fen, 4, c_helpers.evaluate_with_pst, tt, 0, None, None, counters
    )

    assert isinstance(best_move, str)
    assert len(tt) > 0


def test_singular_extensions_integration(tactical_fen):
    """Test singular extensions integrated in search."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    import time

    start = time.time()
    best_move = c_helpers.get_best_move_uci(
        tactical_fen, 5, c_helpers.evaluate_with_pst, tt, 0, killers, history, counters
    )
    elapsed = time.time() - start

    assert isinstance(best_move, str)
    assert elapsed < 15.0  # Should complete in reasonable time


def test_continuation_history_functionality(starting_fen):
    """Test continuation history tracking."""
    tt = c_helpers.TranspositionTable()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    # Multiple searches to build up history
    for depth in [2, 3, 4]:
        c_helpers.get_best_move_uci(
            starting_fen,
            depth,
            c_helpers.evaluate_with_pst,
            tt,
            1,
            None,
            history,
            counters,
        )

    # Should have populated tables
    assert len(tt) > 0
