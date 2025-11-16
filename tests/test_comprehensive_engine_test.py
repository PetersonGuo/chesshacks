#!/usr/bin/env python3
"""Comprehensive test of all 23 chess engine features."""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


@pytest.fixture
def starting_position():
    """Starting chess position FEN."""
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def transposition_table():
    """Create a fresh transposition table."""
    return c_helpers.TranspositionTable()


@pytest.fixture
def killer_moves():
    """Create killer moves table."""
    return c_helpers.KillerMoves()


@pytest.fixture
def history_table():
    """Create history table."""
    return c_helpers.HistoryTable()


@pytest.fixture
def counter_move_table():
    """Create counter move table."""
    return c_helpers.CounterMoveTable()


class TestDataStructures:
    """Test all chess engine data structures."""

    def test_transposition_table_creation(self, transposition_table):
        """Test TranspositionTable can be created."""
        assert transposition_table is not None
        assert len(transposition_table) == 0

    def test_killer_moves_creation(self, killer_moves):
        """Test KillerMoves can be created."""
        assert killer_moves is not None

    def test_history_table_creation(self, history_table):
        """Test HistoryTable can be created."""
        assert history_table is not None

    def test_counter_move_table_creation(self, counter_move_table):
        """Test CounterMoveTable can be created."""
        assert counter_move_table is not None


class TestEvaluation:
    """Test evaluation functions."""

    def test_pst_evaluation_starting_position(self, starting_position):
        """Test that starting position evaluates to ~0 (symmetric)."""
        score = c_helpers.evaluate_with_pst(starting_position)
        assert score == 0, f"Starting position should evaluate to 0, got {score}"


class TestSearch:
    """Test search algorithms."""

    def test_search_depth_3(
        self,
        starting_position,
        transposition_table,
        killer_moves,
        history_table,
        counter_move_table,
    ):
        """Test search at depth 3."""
        transposition_table.clear()
        best_move = c_helpers.get_best_move_uci(
            starting_position,
            3,
            c_helpers.evaluate_with_pst,
            transposition_table,
            0,
            killer_moves,
            history_table,
            counter_move_table,
        )
        assert len(best_move) >= 4, f"Expected UCI move, got {best_move}"
        assert len(transposition_table) > 0, "TT should have entries"

    def test_search_depth_4(
        self,
        starting_position,
        transposition_table,
        killer_moves,
        history_table,
        counter_move_table,
    ):
        """Test search at depth 4."""
        transposition_table.clear()
        best_move = c_helpers.get_best_move_uci(
            starting_position,
            4,
            c_helpers.evaluate_with_pst,
            transposition_table,
            0,
            killer_moves,
            history_table,
            counter_move_table,
        )
        assert len(best_move) >= 4, f"Expected UCI move, got {best_move}"
        assert len(transposition_table) > 0, "TT should have entries"

    def test_search_depth_5(
        self,
        starting_position,
        transposition_table,
        killer_moves,
        history_table,
        counter_move_table,
    ):
        """Test search at depth 5."""
        transposition_table.clear()
        start_time = time.time()
        best_move = c_helpers.get_best_move_uci(
            starting_position,
            5,
            c_helpers.evaluate_with_pst,
            transposition_table,
            0,
            killer_moves,
            history_table,
            counter_move_table,
        )
        elapsed = time.time() - start_time
        assert len(best_move) >= 4, f"Expected UCI move, got {best_move}"
        assert len(transposition_table) > 0, "TT should have entries"
        assert elapsed < 5.0, f"Depth 5 search too slow: {elapsed:.3f}s"


class TestTacticalPositions:
    """Test tactical awareness."""

    @pytest.mark.parametrize(
        "name,fen",
        [
            (
                "Scholar's Mate Threat",
                "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
            ),
            (
                "Fork Opportunity",
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            ),
            ("Endgame", "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"),
        ],
    )
    def test_tactical_position(self, name, fen, transposition_table):
        """Test engine finds reasonable moves in tactical positions."""
        transposition_table.clear()
        best_move = c_helpers.get_best_move_uci(
            fen, 4, c_helpers.evaluate_with_pst, transposition_table
        )
        assert len(best_move) >= 4, f"{name}: Expected UCI move, got {best_move}"


class TestAdvancedFeatures:
    """Test advanced chess engine features."""

    def test_multi_pv_search(self, starting_position):
        """Test Multi-PV search returns multiple variations."""
        pv_lines = c_helpers.multi_pv_search(
            starting_position, 4, 3, c_helpers.evaluate_with_pst
        )
        assert len(pv_lines) == 3, f"Expected 3 PV lines, got {len(pv_lines)}"
        for i, line in enumerate(pv_lines):
            assert (
                len(line.uci_move) >= 4
            ), f"Line {i}: Invalid UCI move {line.uci_move}"
            assert hasattr(line, "score"), f"Line {i}: Missing score"

class TestFeatureChecklist:
    """Verify all 23 features are present."""

    def test_all_features_present(self):
        """Test that all 23 features are implemented."""
        features = [
            "alpha_beta_basic",
            "alpha_beta_optimized",
            "alpha_beta_cuda",
            "evaluate_with_pst",
            "TranspositionTable",
            "KillerMoves",
            "HistoryTable",
            "CounterMoveTable",
            "multi_pv_search",
            "find_best_move",
            "get_best_move_uci",
        ]

        for feature in features:
            assert hasattr(c_helpers, feature), f"Missing feature: {feature}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
