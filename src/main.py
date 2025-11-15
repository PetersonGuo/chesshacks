"""
Chess Engine Main - Game Connection Interface

This module provides the entrypoint and reset functions required by the
chess_manager decorator system for connecting to the game server.
"""

from .utils import chess_manager, GameContext
from chess import Move
import sys
import os

# Add build directory to path so c_helpers can be imported
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if build_path not in sys.path:
    sys.path.insert(0, build_path)

import c_helpers

# Import engine utilities
from . import engine

# Global configuration
SEARCH_DEPTH = 5  # Depth 5 averages ~200-1200ms, safe for 1-minute games
NUM_THREADS = 0  # 0 = auto-detect CPU cores

# Create persistent C++ resources (reused across moves)
transposition_table = c_helpers.TranspositionTable()
killer_moves = c_helpers.KillerMoves()
history_table = c_helpers.HistoryTable()

# Detect CUDA at startup
USE_CUDA = engine.has_cuda()

print("Chess Engine initialized:")
print(f"  - CUDA available: {USE_CUDA}")
if USE_CUDA:
    print(f"  - CUDA info: {engine.get_cuda_status()}")
print(f"  - Search depth: {SEARCH_DEPTH}")
print(f"  - Threads: {NUM_THREADS if NUM_THREADS > 0 else 'auto'}")


@chess_manager.entrypoint
def make_move(ctx: GameContext) -> Move:
    """
    Main entrypoint - called every time the engine needs to make a move.
    Returns a python-chess Move object that is a legal move for the current position.
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
    try:
        best_move = Move.from_uci(best_move_uci)
    except ValueError:
        # Fallback if UCI parsing fails
        print(f"Warning: Failed to parse UCI move '{best_move_uci}'")
        legal_moves = list(ctx.board.generate_legal_moves())
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available")
        best_move = legal_moves[0]

    # For move probabilities, we can use multi-PV to get alternatives
    # For now, assign full probability to the best move
    ctx.logProbabilities({best_move: 1.0})

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
