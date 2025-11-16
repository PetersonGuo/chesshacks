"""
Chess Engine Main - Game Connection Interface

This module provides the entrypoint and reset functions required by the
chess_manager decorator system for connecting to the game server.
"""

import math
import os
from typing import Dict

import chess
from chess import Move

from .native_loader import ensure_c_helpers
from .utils import GameContext, chess_manager

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c_helpers = ensure_c_helpers()

# Import engine utilities
from . import engine

# Global configuration
SEARCH_DEPTH = 4  # Depth 5 averages ~200-1200ms, safe for 1-minute games
NUM_THREADS = 1  # 1 = single-threaded (0 = auto-detect CPU cores) - using 1 to avoid GIL issues
PROBABILITY_TEMPERATURE = 1.0  # Lower = sharper distribution

# Create persistent C++ resources (reused across moves)
transposition_table = c_helpers.TranspositionTable()
killer_moves = c_helpers.KillerMoves()
history_table = c_helpers.HistoryTable()

# Detect CUDA at startup
USE_CUDA = engine.has_cuda()

# Initialize NNUE model if available
NNUE_MODEL_PATH = os.path.join(project_root, "train", "nnue_model", "checkpoints", "best_model.bin")
NNUE_LOADED = False


def load_nnue_model() -> bool:
    """Load the NNUE model if it is available and not already loaded."""
    global NNUE_LOADED

    if NNUE_LOADED:
        return True

    if not os.path.exists(NNUE_MODEL_PATH):
        print(f"NNUE model not found at {NNUE_MODEL_PATH}")
        return False

    try:
        NNUE_LOADED = c_helpers.init_nnue(NNUE_MODEL_PATH)
        if not NNUE_LOADED:
            print(f"Warning: Failed to load NNUE model from {NNUE_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading NNUE model: {e}")
        NNUE_LOADED = False

    return NNUE_LOADED


load_nnue_model()

print("Chess Engine initialized:")
print(f"  - CUDA available: {USE_CUDA}")
if USE_CUDA:
    print(f"  - CUDA info: {engine.get_cuda_status()}")
print(f"  - NNUE loaded: {NNUE_LOADED}")
if NNUE_LOADED:
    print(f"  - NNUE model: {NNUE_MODEL_PATH}")
print(f"  - Search depth: {SEARCH_DEPTH}")
print(f"  - Threads: {NUM_THREADS if NUM_THREADS > 0 else 'auto'}")


def _build_probability_distribution(fen: str) -> Dict[Move, float]:
    """
    Build a lightweight probability distribution using Python-only heuristics.

    The previous implementation relied on the multi-PV search in the C++ engine,
    but that path was prone to native crashes on some environments. To keep the
    devtools stable we approximate the distribution by rewarding captures,
    checks, and castling moves. This is sufficient for debugging/visualization
    without invoking the unstable multi-PV entrypoint.
    """
    board_snapshot = chess.Board(fen)
    legal_moves = list(board_snapshot.generate_legal_moves())
    if not legal_moves:
        raise ValueError("No legal moves available for probability distribution")

    score_map: Dict[Move, float] = {}
    for move in legal_moves:
        score = 1.0
        if board_snapshot.is_capture(move):
            score += 2.0
        if board_snapshot.gives_check(move):
            score += 1.5
        if board_snapshot.is_castling(move):
            score += 0.5
        if board_snapshot.is_en_passant(move):
            score += 0.25
        score_map[move] = score

    return score_map


@chess_manager.entrypoint
def make_move(ctx: GameContext) -> Move:
    """
    Main entrypoint - called every time the engine needs to make a move.
    Returns the chosen Move while logging the probability distribution for UI.
    """
    print(f"Thinking... (depth={SEARCH_DEPTH})")

    # Get current position as FEN
    fen = ctx.board.fen()

    # Use C++ engine to find best move
    # Try with CUDA if available, fall back to CPU on error
    try:
        best_move_uci = c_helpers.get_best_move_uci(
            fen,
            SEARCH_DEPTH,
            engine.nnue_evaluate,
            transposition_table,
            NUM_THREADS,
            killer_moves,
            history_table,
            c_helpers.CounterMoveTable(),  # Create counter move table
        )
    except Exception as e:
        # If CUDA or any C++ error occurs, log and retry with fallback parameters
        print(f"Engine error ({e}), retrying with fallback...")
        best_move_uci = c_helpers.get_best_move_uci(
            fen,
            SEARCH_DEPTH,
            engine.nnue_evaluate,
            transposition_table,
            0,  # Single thread as fallback
            killer_moves,
            history_table,
            c_helpers.CounterMoveTable(),
        )

    print(f"Engine selected: {best_move_uci}")

    # Convert UCI string to python-chess Move object
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    try:
        best_move = Move.from_uci(best_move_uci)
    except ValueError:
        print(f"Warning: Failed to parse UCI move '{best_move_uci}'")
        best_move = legal_moves[0]

    score_map = _build_probability_distribution(ctx.board.fen())
    if not score_map:
        score_map = {best_move: 1.0}
    else:
        score_map[best_move] = score_map.get(best_move, 0.0) + 3.0

    max_score = max(score_map.values())
    temperature = PROBABILITY_TEMPERATURE if PROBABILITY_TEMPERATURE > 0 else 1.0
    exp_scores = {
        move: math.exp((score - max_score) / temperature) for move, score in score_map.items()
    }
    total = sum(exp_scores.values())
    probabilities = {move: value / total for move, value in exp_scores.items()}
    ctx.logProbabilities(probabilities)

    return best_move


@chess_manager.reset
def reset_game(ctx: GameContext):
    """
    Reset handler - called when a new game begins.
    Clears C++ resources for a fresh start.
    """
    global transposition_table, killer_moves, history_table
    transposition_table.clear()
    killer_moves.clear()
    history_table.clear()
    print("New game started - C++ resources cleared")


# For standalone testing
if __name__ == "__main__":
    engine.test_engine()
