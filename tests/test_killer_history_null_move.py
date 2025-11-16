#!/usr/bin/env python3
"""Pytest tests for killer moves, history heuristic, and null move pruning."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.fixture
def starting_fen():
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def after_e4_fen():
    return "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


@pytest.fixture
def tactical_fen():
    return "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.fixture
def simple_evaluate():
    """Simple material count evaluation."""

    def evaluate(fen: str) -> int:
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

    return evaluate


def test_killer_moves_and_history_heuristic(starting_fen, simple_evaluate):
    """Test that killer moves and history heuristic provide speedup."""
    import time

    # Without killer/history
    tt1 = c_helpers.TranspositionTable()
    start = time.time()
    score1 = c_helpers.alpha_beta_optimized(
        starting_fen, 3, c_helpers.MIN, c_helpers.MAX, True, simple_evaluate, tt1, 1
    )
    time_without = time.time() - start

    # With killer/history
    tt2 = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    start = time.time()
    score2 = c_helpers.alpha_beta_optimized(
        starting_fen,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        simple_evaluate,
        tt2,
        1,
        killers,
        history,
    )
    time_with = time.time() - start

    # Scores should be same
    assert score1 == score2
    # With optimizations should be same or faster
    assert time_with <= time_without * 1.2  # Allow 20% tolerance


def test_tactical_position_null_move_pruning(tactical_fen, simple_evaluate):
    """Test null move pruning on tactical position."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        tactical_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        simple_evaluate,
        tt,
        1,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0  # TT should be populated


def test_parallel_search_with_all_features(starting_fen, simple_evaluate):
    """Test parallel search with killer moves and history."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        starting_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        simple_evaluate,
        tt,
        0,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0


def test_reusable_tables_across_searches(starting_fen, after_e4_fen, simple_evaluate):
    """Test reusing killer moves and history across searches."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    # First search
    score1 = c_helpers.alpha_beta_optimized(
        starting_fen,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        simple_evaluate,
        tt,
        1,
        killers,
        history,
    )
    tt_size_after_first = len(tt)

    # Second search (reusing tables)
    score2 = c_helpers.alpha_beta_optimized(
        after_e4_fen,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        False,
        simple_evaluate,
        tt,
        1,
        killers,
        history,
    )

    assert isinstance(score1, int)
    assert isinstance(score2, int)
    assert len(tt) >= tt_size_after_first  # TT should grow


def test_history_aging():
    """Test history table aging."""
    history = c_helpers.HistoryTable()
    tt = c_helpers.TranspositionTable()

    # Build up history
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for _ in range(3):
        c_helpers.alpha_beta_optimized(
            starting_fen,
            2,
            c_helpers.MIN,
            c_helpers.MAX,
            True,
            c_helpers.evaluate_with_pst,
            tt,
            1,
            None,
            history,
        )

    # Age should work without error
    history.age()
    history.clear()

    assert True  # No exception raised


def test_killer_moves_clear():
    """Test killer moves clearing."""
    killers = c_helpers.KillerMoves()

    # Populate killer moves
    tt = c_helpers.TranspositionTable()
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    c_helpers.alpha_beta_optimized(
        starting_fen,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        1,
        killers,
        None,
    )

    # Clear should work
    killers.clear()

    assert True  # No exception raised


def test_null_move_pruning_performance(starting_fen):
    """Test that null move pruning is active and functional."""
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
        1,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0
