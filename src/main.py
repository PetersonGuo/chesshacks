from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import sys
import os

# Add build directory to path so c_helpers can be imported
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if build_path not in sys.path:
    sys.path.insert(0, build_path)

import c_helpers

# Import CUDA checker utility
try:
    from .cuda_check import is_cuda_available, get_cuda_info

    CUDA_CHECKER_AVAILABLE = True
except ImportError:
    CUDA_CHECKER_AVAILABLE = False


# Detect CUDA availability
def has_cuda():
    """Check if CUDA is available on this system"""
    # First check via C++ bindings (if compiled with CUDA)
    try:
        if c_helpers.is_cuda_available():
            return True
    except AttributeError:
        pass

    # Then check via PyTorch if available
    if CUDA_CHECKER_AVAILABLE:
        cuda_available, _ = is_cuda_available()
        if cuda_available:
            return True

    # Finally check nvidia-smi as fallback
    try:
        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=1)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_cuda_status():
    """Get detailed CUDA status information"""
    # Try C++ method first
    try:
        cpp_info = c_helpers.get_cuda_info()
        return cpp_info
    except AttributeError:
        pass

    # Try Python method
    if CUDA_CHECKER_AVAILABLE:
        cuda_available, msg = is_cuda_available()
        if cuda_available:
            info = get_cuda_info()
            if info:
                return f"{info['device_count']} GPU(s): {info['devices'][0]['name']}"
        return msg

    return "CUDA detection not available"


# Global configuration
USE_CUDA = has_cuda()
SEARCH_DEPTH = 4  # Adjust based on time constraints
NUM_THREADS = 0  # 0 = auto-detect CPU cores

# Create persistent C++ resources (reused across moves)
transposition_table = c_helpers.TranspositionTable()
killer_moves = c_helpers.KillerMoves()
history_table = c_helpers.HistoryTable()

print(f"Chess Engine initialized:")
print(f"  - CUDA available: {USE_CUDA}")
if USE_CUDA:
    print(f"  - CUDA info: {get_cuda_status()}")
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
        move: weight / total_weight for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Clear C++ resources for fresh start
    global transposition_table, killer_moves, history_table
    transposition_table.clear()
    killer_moves.clear()
    history_table.clear()
    print(f"New game started - C++ resources cleared")


def nnue_evaluate(fen: str) -> int:
    """
    Placeholder evaluation function
    TODO: Replace with actual NNUE model
    Currently returns simple material count
    """
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
    score = 0
    for char in fen.split()[0]:
        if char in piece_values:
            score += piece_values[char]
    return score


def alpha_beta_basic(
    fen: str,
    depth: int,
    alpha: int = None,
    beta: int = None,
    maximizing_player: bool = None,
    evaluate=None,
):
    """
    Call C++ alpha_beta_basic function - bare bones alpha-beta with no optimizations

    Args:
        fen: FEN string of the position
        depth: Search depth
        alpha: Alpha value (defaults to MIN)
        beta: Beta value (defaults to MAX)
        maximizing_player: True if maximizing player (defaults to True if FEN indicates white)
        evaluate: Evaluation function (defaults to nnue_evaluate)

    Returns:
        Evaluation score
    """
    if alpha is None:
        alpha = c_helpers.MIN
    if beta is None:
        beta = c_helpers.MAX
    if maximizing_player is None:
        # Determine from FEN (second part should be 'w' or 'b')
        parts = fen.split()
        maximizing_player = len(parts) > 1 and parts[1] == "w"
    if evaluate is None:
        evaluate = nnue_evaluate

    return c_helpers.alpha_beta_basic(
        fen, depth, alpha, beta, maximizing_player, evaluate
    )


def alpha_beta_optimized(
    fen: str,
    depth: int,
    alpha: int = None,
    beta: int = None,
    maximizing_player: bool = None,
    evaluate=None,
    tt=None,
    num_threads: int = 0,
    killers=None,
    history=None,
):
    """
    Call C++ alpha_beta_optimized function - full optimizations (TT, move ordering, etc.)

    Args:
        fen: FEN string of the position
        depth: Search depth
        alpha: Alpha value (defaults to MIN)
        beta: Beta value (defaults to MAX)
        maximizing_player: True if maximizing player (defaults to True if FEN indicates white)
        evaluate: Evaluation function (defaults to nnue_evaluate)
        tt: TranspositionTable instance (optional, creates new one if None)
        num_threads: Number of threads (0 = auto, 1 = sequential)
        killers: KillerMoves instance (optional)
        history: HistoryTable instance (optional)

    Returns:
        Evaluation score
    """
    if alpha is None:
        alpha = c_helpers.MIN
    if beta is None:
        beta = c_helpers.MAX
    if maximizing_player is None:
        # Determine from FEN
        parts = fen.split()
        maximizing_player = len(parts) > 1 and parts[1] == "w"
    if evaluate is None:
        evaluate = nnue_evaluate

    return c_helpers.alpha_beta_optimized(
        fen,
        depth,
        alpha,
        beta,
        maximizing_player,
        evaluate,
        tt,
        num_threads,
        killers,
        history,
    )


def alpha_beta_cuda(
    fen: str,
    depth: int,
    alpha: int = None,
    beta: int = None,
    maximizing_player: bool = None,
    evaluate=None,
    tt=None,
    killers=None,
    history=None,
):
    """
    Call C++ alpha_beta_cuda function - CUDA-accelerated search (falls back to optimized)

    Args:
        fen: FEN string of the position
        depth: Search depth
        alpha: Alpha value (defaults to MIN)
        beta: Beta value (defaults to MAX)
        maximizing_player: True if maximizing player (defaults to True if FEN indicates white)
        evaluate: Evaluation function (defaults to nnue_evaluate)
        tt: TranspositionTable instance (optional)
        killers: KillerMoves instance (optional)
        history: HistoryTable instance (optional)

    Returns:
        Evaluation score
    """
    if alpha is None:
        alpha = c_helpers.MIN
    if beta is None:
        beta = c_helpers.MAX
    if maximizing_player is None:
        # Determine from FEN
        parts = fen.split()
        maximizing_player = len(parts) > 1 and parts[1] == "w"
    if evaluate is None:
        evaluate = nnue_evaluate

    return c_helpers.alpha_beta_cuda(
        fen, depth, alpha, beta, maximizing_player, evaluate, tt, killers, history
    )


def pgn_to_fen(pgn: str) -> str:
    """
    Convert PGN string to FEN string using C++ function.
    
    Args:
        pgn: PGN (Portable Game Notation) string containing game moves
    
    Returns:
        FEN string representing the final position after all moves
    """
    return c_helpers.pgn_to_fen(pgn)


def search_position(fen: str, depth: int = SEARCH_DEPTH, tt=None, killers=None, history=None) -> int:
    """
    Search a position using the best available engine
    Returns the evaluation score

    Args:
        fen: FEN string of the position
        depth: Search depth
        tt: TranspositionTable instance (uses global if None)
        killers: KillerMoves instance (uses global if None)
        history: HistoryTable instance (uses global if None)

    Returns:
        Evaluation score
    """
    global transposition_table, killer_moves, history_table

    if tt is None:
        tt = transposition_table
    if killers is None:
        killers = killer_moves
    if history is None:
        history = history_table

    if USE_CUDA:
        # Use CUDA-accelerated search if available
        return alpha_beta_cuda(
            fen, depth, evaluate=nnue_evaluate, tt=tt, killers=killers, history=history
        )
    else:
        # Use optimized CPU search with all features
        return alpha_beta_optimized(
            fen,
            depth,
            evaluate=nnue_evaluate,
            tt=tt,
            num_threads=NUM_THREADS,
            killers=killers,
            history=history,
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
