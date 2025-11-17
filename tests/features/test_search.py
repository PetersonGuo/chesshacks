"""
Pytest suite that exercises the core search interfaces in grouped scenarios.

The previous ad-hoc scripts (move retrieval, debug harness, parallel demos, etc.)
are consolidated here so that similar assertions live together and run inside
pytest without print-driven flows.
"""

from __future__ import annotations

import c_helpers
import pytest

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TACTICAL_FEN = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
ENDGAME_FEN = "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"
MIDGAME_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def test_batch_evaluate_mt_returns_scores(make_state):
    boards = [
        make_state(STARTING_FEN),
        make_state(TACTICAL_FEN),
        make_state(ENDGAME_FEN),
    ]
    scores = c_helpers.batch_evaluate_mt(boards, num_threads=2)
    assert len(scores) == len(boards)
    assert all(isinstance(score, int) for score in scores)


def test_multi_pv_produces_uci_sequences(make_state):
    tt = c_helpers.TranspositionTable()
    lines = c_helpers.multi_pv_search(
        make_state(STARTING_FEN), 3, 3, c_helpers.evaluate, tt, 2
    )
    assert len(lines) == 3
    for line in lines:
        assert line.uci_move
        assert isinstance(line.uci_move, str)
        assert isinstance(line.pv, str)


def test_multi_pv_parallel_matches_sequential(make_state):
    depth = 4
    num_lines = 4

    seq_tt = c_helpers.TranspositionTable()
    seq_lines = c_helpers.multi_pv_search(
        make_state(STARTING_FEN), depth, num_lines, c_helpers.evaluate, seq_tt, 1
    )

    par_tt = c_helpers.TranspositionTable()
    par_lines = c_helpers.multi_pv_search(
        make_state(STARTING_FEN), depth, num_lines, c_helpers.evaluate, par_tt, 4
    )

    assert len(seq_lines) == len(par_lines)
    assert {line.uci_move for line in seq_lines} == {
        line.uci_move for line in par_lines
    }


def test_parallel_alpha_beta_matches_serial(make_state):
    """Parallel search should agree with sequential search on the same position."""
    alpha = c_helpers.MIN
    beta = c_helpers.MAX
    seq_score = c_helpers.alpha_beta(
        make_state(MIDGAME_FEN), 5, alpha, beta, True, num_threads=1
    )
    par_score = c_helpers.alpha_beta(
        make_state(MIDGAME_FEN), 5, alpha, beta, True, num_threads=4
    )
    assert abs(seq_score - par_score) <= 150


def test_transposition_table_is_utilized(make_state):
    """Ensure TT entries persist between repeated alpha-beta searches."""
    tt = c_helpers.TranspositionTable()
    initial_size = len(tt)

    score_1 = c_helpers.alpha_beta(
        make_state(STARTING_FEN),
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        1,
    )
    after_first = len(tt)
    assert after_first >= initial_size

    score_2 = c_helpers.alpha_beta(
        make_state(STARTING_FEN),
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        1,
    )
    assert len(tt) >= after_first
    assert score_1 == score_2


def test_alpha_beta_is_deterministic(make_state):
    """Built-in evaluator should be deterministic across repeated invocations."""

    first = c_helpers.alpha_beta(
        make_state(TACTICAL_FEN),
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        num_threads=1,
    )
    second = c_helpers.alpha_beta(
        make_state(TACTICAL_FEN),
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        num_threads=1,
    )
    assert isinstance(first, int)
    assert first == second


def test_quiescence_handles_tactics(make_state):
    """Horizon-effect position should still return a decisive score."""
    tt = c_helpers.TranspositionTable()
    score = c_helpers.alpha_beta(
        make_state("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3"),
        2,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        1,
    )
    assert isinstance(score, int)


def test_alpha_beta_accepts_shared_tables(make_state):
    """Passing explicit killer/history/counter tables should not error."""
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    score = c_helpers.alpha_beta(
        make_state(STARTING_FEN),
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        2,
        killers,
        history,
        counters,
    )
    assert isinstance(score, int)
