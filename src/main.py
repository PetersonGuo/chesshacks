"""
Chess Engine Main - Game Connection Interface

Provides the entrypoint/reset hooks required by the chess_manager decorator.
All internal state stays in Virgo bitboards; FEN is used only at the API edges.
"""

import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import chess
from chess import Move

import engine
from env_manager import get_env_config
from native_loader import ensure_c_helpers
from utils import GameContext, chess_manager

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_CONFIG = get_env_config()


# Optional NNUE model (falls back silently if file missing)
NNUE_MODEL_PATH = os.path.abspath(ENV_CONFIG.nnue_model_path)
NNUE_LOADED = False


def load_nnue_model() -> bool:
    global NNUE_LOADED

    if NNUE_LOADED:
        return True
    model_path = Path(NNUE_MODEL_PATH)
    if not model_path.exists():
        print(f"NNUE model file not found at {model_path}")
        return False
    try:
        NNUE_LOADED = c_helpers.init_nnue(str(model_path))
    except Exception as exc:
        print(f"NNUE load failed ({exc})")
        NNUE_LOADED = False

    return NNUE_LOADED


c_helpers = ensure_c_helpers()

# Global configuration
SEARCH_DEPTH = ENV_CONFIG.search_depth
c_helpers.set_max_search_depth(SEARCH_DEPTH)
NUM_THREADS = ENV_CONFIG.search_threads
PROBABILITY_TEMPERATURE = 200.0

# Persistent native resources
transposition_table = c_helpers.TranspositionTable()
killer_moves = c_helpers.KillerMoves()
history_table = c_helpers.HistoryTable()
counter_move_table = c_helpers.CounterMoveTable()

# Capability detection
USE_CUDA = engine.has_cuda()

load_nnue_model()


print("Chess Engine initialized:")
print(f"  - CUDA available: {USE_CUDA}")
if USE_CUDA:
    print(f"  - CUDA info: {engine.get_cuda_status()}")
print(f"  - NNUE loaded: {NNUE_LOADED}")
if NNUE_LOADED:
    print(f"  - NNUE model: {NNUE_MODEL_PATH}")
else:
    print(f"  - NNUE model path (pending): {NNUE_MODEL_PATH}")
print(f"  - Search depth: {SEARCH_DEPTH}")
print(f"  - Threads: {NUM_THREADS if NUM_THREADS > 0 else 'auto'}")


def _fallback_probability_distribution(board: chess.Board) -> Dict[Move, float]:
    score_map: Dict[Move, float] = {}
    for move in board.generate_legal_moves():
        score = 1.0
        if board.is_capture(move):
            score += 2.0
        board.push(move)
        if board.is_check():
            score += 1.5
        board.pop()
        if board.is_castling(move):
            score += 0.5
        if move.promotion:
            score += 1.0
        score_map[move] = score
    return score_map


def _search_with_probabilities(
    board: chess.Board, bitboard_state: c_helpers.BitboardState
) -> tuple[Optional[Move], Dict[Move, float]]:
    """
    Run multi-PV search once to obtain both the best move and probability map.
    Returns (best_move, score_map).
    """
    board_snapshot = board.copy()
    legal_moves = set(board_snapshot.generate_legal_moves())
    if not legal_moves:
        raise ValueError("No legal moves available for probability distribution")

    max_lines = min(5, len(legal_moves))
    score_map: Dict[Move, float] = {}
    best_move: Optional[Move] = None

    try:
        pv_lines = c_helpers.multi_pv_search_state(
            bitboard_state,
            SEARCH_DEPTH,
            max_lines,
            c_helpers.evaluate,
            transposition_table,
            NUM_THREADS,
            killer_moves,
            history_table,
            counter_move_table,
        )

        for line in pv_lines:
            try:
                move = Move.from_uci(line.uci_move)
            except ValueError:
                continue
            if move in legal_moves:
                score_map[move] = line.score
                if best_move is None:
                    best_move = move
    except Exception as exc:
        print(f"multi_pv_search failed ({exc}); falling back to heuristics")
        return None, _fallback_probability_distribution(board_snapshot)

    if not score_map:
        return None, _fallback_probability_distribution(board_snapshot)
    return best_move, score_map


def _search_best_move(bitboard_state: c_helpers.BitboardState) -> str:
    """Search for the best move using the fastest native evaluation path."""
    return c_helpers.get_best_move_uci_state(
        bitboard_state,
        SEARCH_DEPTH,
        c_helpers.evaluate,
        transposition_table,
        NUM_THREADS,
        killer_moves,
        history_table,
        counter_move_table,
    )


@chess_manager.entrypoint
def make_move(ctx: GameContext) -> Dict[Move, float]:
    """
    Main entrypoint – returns a probability distribution over legal moves.
    """
    print(f"Thinking... (depth={SEARCH_DEPTH})")

    bitboard_state = engine._board_to_bitboard_state(ctx.board)

    best_move: Optional[Move] = None
    score_map: Dict[Move, float] = {}
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Run a single multi-PV search to get both best move and probabilities
    best_move, score_map = _search_with_probabilities(ctx.board, bitboard_state)

    if best_move is None:
        try:
            best_move_uci = _search_best_move(bitboard_state)
        except Exception as exc:
            print(f"Engine error ({exc}), retrying with fallback…")
            best_move_uci = c_helpers.get_best_move_uci_builtin_state(
                bitboard_state,
                SEARCH_DEPTH,
                transposition_table,
                0,
                killer_moves,
                history_table,
                counter_move_table,
            )
    try:
        best_move = Move.from_uci(best_move_uci)
    except ValueError:
        legal_moves = list(ctx.board.generate_legal_moves())
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available")
        print(f"Warning: Failed to parse UCI move '{best_move_uci}'")
        best_move = legal_moves[0]
        best_move_uci = best_move.uci()
    else:
        best_move_uci = best_move.uci()

    print(f"Engine selected: {best_move_uci}")

    if not score_map:
        score_map = {best_move: 1.0}
    elif best_move not in score_map:
        score_map[best_move] = 0.0

    max_score = max(score_map.values())
    exp_scores = {
        move: math.exp((score - max_score) / PROBABILITY_TEMPERATURE)
        for move, score in score_map.items()
    }
    total = sum(exp_scores.values())
    probabilities = {move: value / total for move, value in exp_scores.items()}
    ctx.logProbabilities(probabilities)

    return best_move


@chess_manager.reset
def reset_game(ctx: GameContext):
    """
    Reset handler – clears native resources between games.
    """
    global transposition_table, killer_moves, history_table, counter_move_table
    transposition_table.clear()
    killer_moves.clear()
    history_table.clear()
    counter_move_table.clear()
    print("New game started - C++ resources cleared")


# For standalone testing
if __name__ == "__main__":
    engine.test_engine()
