"""
Chess Engine Utilities

This module contains helper functions and utilities for the chess engine.
These are not directly needed for the game connection but provide useful
functionality for testing and development.
"""

import sys
import os
import time

# Add build directory to path so c_helpers can be imported
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if build_path not in sys.path:
    sys.path.insert(0, build_path)

import c_helpers


def has_cuda():
    """Check if CUDA is available using torch"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def get_cuda_status():
    """Get detailed CUDA status information using torch"""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return f"{device_count} GPU(s): {device_name}"
        return "CUDA not available"
    except ImportError:
        return "PyTorch not installed"
    except Exception as e:
        return f"CUDA detection error: {e}"


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


def search_position(
    fen: str,
    depth: int,
    use_cuda: bool = False,
    num_threads: int = 0,
    tt=None,
    killers=None,
    history=None,
) -> int:
    """
    Search a position using the best available engine
    Returns the evaluation score

    Args:
        fen: FEN string of the position
        depth: Search depth
        use_cuda: Whether to use CUDA acceleration
        num_threads: Number of threads (0 = auto)
        tt: TranspositionTable instance (creates new if None)
        killers: KillerMoves instance (creates new if None)
        history: HistoryTable instance (creates new if None)

    Returns:
        Evaluation score
    """
    if use_cuda:
        # Try CUDA-accelerated search, fall back to CPU if it fails
        try:
            return alpha_beta_cuda(
                fen,
                depth,
                evaluate=nnue_evaluate,
                tt=tt,
                killers=killers,
                history=history,
            )
        except Exception as e:
            # CUDA failed, fall back to CPU
            print(f"CUDA search failed ({e}), falling back to CPU")
            return alpha_beta_optimized(
                fen,
                depth,
                evaluate=nnue_evaluate,
                tt=tt,
                num_threads=num_threads,
                killers=killers,
                history=history,
            )
    else:
        # Use optimized CPU search with all features
        return alpha_beta_optimized(
            fen,
            depth,
            evaluate=nnue_evaluate,
            tt=tt,
            num_threads=num_threads,
            killers=killers,
            history=history,
        )


def test_engine():
    """Test the search engine"""
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    use_cuda = has_cuda()

    print("\nTesting search engine...")
    print(f"Position: {starting_fen}")
    print(f"Using: {'CUDA' if use_cuda else 'CPU optimized'}")

    # Create test resources
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    start_time = time.time()
    result = search_position(
        starting_fen,
        depth=4,
        use_cuda=use_cuda,
        tt=tt,
        killers=killers,
        history=history,
    )
    elapsed = time.time() - start_time

    print(f"Search result: {result}")
    print(f"Time: {elapsed:.3f}s")
    print(f"TT entries: {tt.size()}")
    print("Features active:")
    print("  ✓ Transposition table")
    print("  ✓ Move ordering (TT + MVV-LVA + promotions)")
    print("  ✓ Quiescence search")
    print("  ✓ Iterative deepening")
    if not use_cuda:
        print("  ✓ Parallel search")


if __name__ == "__main__":
    test_engine()
