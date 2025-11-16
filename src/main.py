"""
Chess Engine Main - Game Connection Interface

This module provides the entrypoint and reset functions required by the
chess_manager decorator system for connecting to the game server.
"""

import math
import os
import subprocess
import sys
from typing import Dict

import chess
from chess import Move

from .utils import GameContext, chess_manager

# Add paths for c_helpers module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_path = os.path.join(project_root, "build")
if build_path not in sys.path:
    sys.path.insert(0, build_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import c_helpers, build if necessary
try:
    import c_helpers
except ModuleNotFoundError:
    print("=" * 80)
    print("C++ module not found - triggering automatic build...")
    print("=" * 80)

    build_script = os.path.join(project_root, "build.sh")
    if os.path.exists(build_script):
        try:
            # Run build script
            result = subprocess.run(
                ["/bin/bash", build_script],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)

            # Try import again
            import c_helpers

            print("✓ Build successful - module imported")
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            raise
        except ModuleNotFoundError:
            print("Build completed but module still not found.")
            print("Please run: ./build.sh")
            raise
    else:
        print(f"Build script not found at: {build_script}")
        print("Please run: ./build.sh from project root")
        raise

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


def _build_probability_distribution(fen: str) -> Dict[Move, float]:
    """Use multi-PV search to construct a probability distribution over moves."""
    board_snapshot = chess.Board(fen)
    legal_moves = set(board_snapshot.generate_legal_moves())
    if not legal_moves:
        raise ValueError("No legal moves available for probability distribution")

    max_lines = min(5, len(legal_moves))
    score_map: Dict[Move, float] = {}

    try:
        pv_lines = c_helpers.multi_pv_search(
            fen,
            SEARCH_DEPTH,
            max_lines,
            engine.nnue_evaluate,
            transposition_table,
            NUM_THREADS,
            killer_moves,
            history_table,
            c_helpers.CounterMoveTable(),
        )

        for line in pv_lines:
            try:
                move = Move.from_uci(line.uci_move)
            except ValueError:
                continue
            if move in legal_moves:
                score_map[move] = line.score
    except Exception as exc:
        print(f"multi_pv_search failed ({exc}); falling back to deterministic move")

    return score_map


@chess_manager.entrypoint
def make_move(ctx: GameContext) -> dict[Move, float]:
    """
    Main entrypoint - called every time the engine needs to make a move.
    Returns a probability distribution over legal moves (dict[Move, float]).
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
    if best_move not in score_map:
        score_map[best_move] = 0.0

    if not score_map:
        score_map = {best_move: 1.0}

    max_score = max(score_map.values())
    exp_scores = {
        move: math.exp((score - max_score) / 200.0) for move, score in score_map.items()
    }
    total = sum(exp_scores.values())
    probabilities = {move: value / total for move, value in exp_scores.items()}
    ctx.logProbabilities(probabilities)

    return probabilities


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
