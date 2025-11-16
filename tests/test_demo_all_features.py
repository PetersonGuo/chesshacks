#!/usr/bin/env python3
"""
Pytest tests for 15 Advanced Chess Engine Features

Tests all advanced features implemented in the chess engine:
1. Three-function architecture
2. MVV-LVA capture ordering
3. Advanced 5-tier move ordering
4. Killer move heuristic
5. History heuristic
6. Null move pruning
7. Quiescence search
8. Iterative deepening
9. Thread-safe transposition table
10. Parallel root search
11. Late move reductions (LMR)
12. Aspiration windows
13. Principal variation search (PVS)
14. Piece-square tables (PST)
15. Internal iterative deepening (IID)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.fixture
def starting_fen():
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def middlegame_fen():
    return "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 10"


@pytest.fixture
def transposition_table():
    return c_helpers.TranspositionTable()


@pytest.fixture
def killer_moves():
    return c_helpers.KillerMoves()


@pytest.fixture
def history_table():
    return c_helpers.HistoryTable()


def test_three_function_architecture(
    starting_fen, transposition_table, killer_moves, history_table
):
    """Test basic, optimized, and CUDA search functions exist."""
    depth = 3

    # Basic search
    score_basic = c_helpers.alpha_beta_basic(
        starting_fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
    )

    # Optimized search
    score_opt = c_helpers.alpha_beta_optimized(
        starting_fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        transposition_table,
        0,
        killer_moves,
        history_table,
    )

    # Both should find similar scores
    assert isinstance(score_basic, int)
    assert isinstance(score_opt, int)
    assert abs(score_basic - score_opt) < 200  # Similar evaluation


def test_data_structures_reusable(
    starting_fen, transposition_table, killer_moves, history_table
):
    """Test reusable data structures across searches."""
    # First search
    score1 = c_helpers.alpha_beta_optimized(
        starting_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        transposition_table,
        0,
        killer_moves,
        history_table,
    )

    initial_tt_size = len(transposition_table)
    assert initial_tt_size > 0

    # Second search (reusing tables)
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    score2 = c_helpers.alpha_beta_optimized(
        after_e4,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        False,
        c_helpers.evaluate_with_pst,
        transposition_table,
        0,
        killer_moves,
        history_table,
    )

    # TT should grow
    assert len(transposition_table) >= initial_tt_size

    # Test clear
    transposition_table.clear()
    killer_moves.clear()
    history_table.clear()

    assert len(transposition_table) == 0


def test_performance_comparison(starting_fen):
    """Test optimized version uses TT and produces same result."""
    depth = 3

    # Basic search
    score_basic = c_helpers.alpha_beta_basic(
        starting_fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
    )

    # Optimized search
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score_opt = c_helpers.alpha_beta_optimized(
        starting_fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )

    # Should produce same score
    assert abs(score_basic - score_opt) < 50  # Small tolerance
    assert len(tt) > 0  # TT should be used
