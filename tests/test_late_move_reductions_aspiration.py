#!/usr/bin/env python3
"""Pytest tests for Late Move Reductions and Aspiration Windows."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.fixture
def starting_fen():
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def middlegame_fen():
    return "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.fixture
def tactical_fen():
    return "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6"


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


def test_lmr_performance(middlegame_fen, simple_evaluate):
    """Test Late Move Reductions on middlegame position."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        middlegame_fen,
        5,
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
    assert len(tt) > 0


def test_aspiration_windows(starting_fen, simple_evaluate):
    """Test aspiration windows with iterative deepening."""
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
        1,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0


def test_tactical_position_lmr(tactical_fen, simple_evaluate):
    """Test LMR on tactical position with many candidate moves."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        tactical_fen,
        5,
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
    assert len(tt) > 0


def test_parallel_search_with_lmr_aspiration(middlegame_fen, simple_evaluate):
    """Test parallel search with LMR and aspiration windows."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        middlegame_fen,
        5,
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


def test_deep_search_performance(starting_fen, simple_evaluate):
    """Test depth 6 with all optimizations."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_optimized(
        starting_fen,
        6,
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


def test_lmr_with_pst_evaluation(middlegame_fen):
    """Test LMR with piece-square table evaluation."""
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
        1,
        killers,
        history,
    )

    assert isinstance(score, int)
    assert len(tt) > 0


def test_aspiration_window_accuracy(starting_fen):
    """Test aspiration windows produce correct scores."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    # Search with aspiration windows (depth >= 3)
    score1 = c_helpers.alpha_beta_optimized(
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

    # Clear and search again
    tt.clear()
    killers.clear()
    history.clear()

    score2 = c_helpers.alpha_beta_optimized(
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

    # Should get same score
    assert score1 == score2
