#!/usr/bin/env python3
"""Pytest tests for Principal Variation Search and Piece-Square Tables."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.fixture
def starting_fen():
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def centralized_knight_fen():
    return "rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"


@pytest.fixture
def italian_fen():
    return "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.fixture
def middlegame_fen():
    return "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 10"


def test_pst_evaluation_starting_position(starting_fen):
    """Test PST evaluation on starting position."""
    eval_score = c_helpers.evaluate_with_pst(starting_fen)

    # Starting position should be close to 0 due to symmetry
    assert abs(eval_score) < 100
    assert isinstance(eval_score, int)


def test_pst_evaluation_centralized_knight(centralized_knight_fen):
    """Test PST evaluation with centralized knight."""
    eval_score = c_helpers.evaluate_with_pst(centralized_knight_fen)

    # White should have positional advantage
    assert isinstance(eval_score, int)


def test_pst_evaluation_italian_game(italian_fen):
    """Test PST evaluation on Italian Game position."""
    eval_score = c_helpers.evaluate_with_pst(italian_fen)

    # Developed pieces should have positional bonuses
    assert isinstance(eval_score, int)


def test_pst_vs_material_comparison(starting_fen, italian_fen):
    """Test PST adds positional evaluation beyond material."""
    # Starting position
    start_pst = c_helpers.evaluate_with_pst(starting_fen)

    # Italian Game
    italian_pst = c_helpers.evaluate_with_pst(italian_fen)

    # Both should return valid scores
    assert isinstance(start_pst, int)
    assert isinstance(italian_pst, int)


def test_pvs_performance_depth_5(middlegame_fen):
    """Test PVS performance at depth 5."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        middlegame_fen,
        5,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0


def test_pvs_performance_depth_6(middlegame_fen):
    """Test PVS performance at depth 6."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    import time

    start = time.time()
    score = c_helpers.alpha_beta_optimized(
        middlegame_fen,
        6,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )
    elapsed = time.time() - start

    assert isinstance(score, int)
    assert len(tt) > 0
    assert elapsed < 30.0  # Should complete in reasonable time


def test_pvs_null_window_efficiency(starting_fen):
    """Test PVS null window searches are efficient."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    import time

    start = time.time()
    score = c_helpers.alpha_beta_optimized(
        starting_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )
    elapsed = time.time() - start

    assert isinstance(score, int)
    assert len(tt) > 0
    assert elapsed < 5.0  # Should be fast


def test_pst_symmetry(starting_fen):
    """Test PST evaluation is symmetric for starting position."""
    eval_score = c_helpers.evaluate_with_pst(starting_fen)

    # Starting position should be nearly equal (small PST bonuses allowed)
    assert abs(eval_score) < 50


def test_pvs_with_transposition_table(starting_fen):
    """Test PVS uses transposition table effectively."""
    tt = c_helpers.TranspositionTable()

    score1 = c_helpers.alpha_beta_optimized(
        starting_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        1,
    )

    tt_size = len(tt)
    assert tt_size > 0

    # Second search with same TT should be faster
    score2 = c_helpers.alpha_beta_optimized(
        starting_fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        1,
    )

    assert score1 == score2
    assert len(tt) == tt_size  # TT shouldn't grow much


def test_pst_different_positions():
    """Test PST gives different evaluations for different positions."""
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen2 = "rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
    fen3 = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

    eval1 = c_helpers.evaluate_with_pst(fen1)
    eval2 = c_helpers.evaluate_with_pst(fen2)
    eval3 = c_helpers.evaluate_with_pst(fen3)

    # All should return valid scores
    assert isinstance(eval1, int)
    assert isinstance(eval2, int)
    assert isinstance(eval3, int)

    # Centralized knight should be better than starting position
    assert eval2 > eval1
