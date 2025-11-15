from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import sys
import os

# Add build directory to path so c_helpers can be imported
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
if build_path not in sys.path:
    sys.path.insert(0, build_path)

import c_helpers

# Detect CUDA availability
def has_cuda():
    """Check if CUDA is available on this system"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=1)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

# Global configuration
USE_CUDA = has_cuda()
SEARCH_DEPTH = 4  # Adjust based on time constraints
NUM_THREADS = 0  # 0 = auto-detect CPU cores

# Create persistent transposition table (reused across moves)
transposition_table = c_helpers.TranspositionTable()

print(f"Chess Engine initialized:")
print(f"  - CUDA available: {USE_CUDA}")
print(f"  - Search depth: {SEARCH_DEPTH}")
print(f"  - Threads: {NUM_THREADS if NUM_THREADS > 0 else 'auto'}")

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Clear transposition table for fresh start
    global transposition_table
    transposition_table.clear()
    print(f"New game started - TT cleared")


def nnue_evaluate(fen: str) -> int:
    """
    Placeholder evaluation function
    TODO: Replace with actual NNUE model
    Currently returns simple material count
    """
    piece_values = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': 0
    }
    score = 0
    for char in fen.split()[0]:
        if char in piece_values:
            score += piece_values[char]
    return score


def search_position(fen: str, depth: int = SEARCH_DEPTH) -> int:
    """
    Search a position using the best available engine
    Returns the evaluation score
    """
    global transposition_table
    
    if USE_CUDA:
        # Use CUDA-accelerated search if available
        return c_helpers.alpha_beta_cuda(
            fen, 
            depth, 
            c_helpers.MIN, 
            c_helpers.MAX, 
            True,  # Assuming white to move, adjust based on FEN
            nnue_evaluate,
            transposition_table
        )
    else:
        # Use optimized CPU search with all features
        return c_helpers.alpha_beta_optimized(
            fen,
            depth,
            c_helpers.MIN,
            c_helpers.MAX,
            True,  # Assuming white to move, adjust based on FEN
            nnue_evaluate,
            transposition_table,
            NUM_THREADS
        )


def main():
    """Test the search engine"""
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"\nTesting search engine...")
    print(f"Position: {starting_fen}")
    print(f"Using: {'CUDA' if USE_CUDA else 'CPU optimized'}")
    
    start_time = time.time()
    result = search_position(starting_fen, depth=4)
    elapsed = time.time() - start_time
    
    print(f"Search result: {result}")
    print(f"Time: {elapsed:.3f}s")
    print(f"TT entries: {len(transposition_table)}")
    print(f"Features active:")
    print(f"  ✓ Transposition table")
    print(f"  ✓ Move ordering (TT + MVV-LVA + promotions)")
    print(f"  ✓ Quiescence search")
    print(f"  ✓ Iterative deepening")
    if not USE_CUDA and NUM_THREADS != 1:
        print(f"  ✓ Parallel search")


if __name__ == "__main__":
    main()
