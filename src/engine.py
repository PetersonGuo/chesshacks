"""
Chess Engine Utilities

This module contains helper functions and utilities for the chess engine.
These are not directly needed for the game connection but provide useful
functionality for testing and development.
"""

import time

import chess

from .env_manager import get_env_config
from .native_loader import ensure_c_helpers

c_helpers = ensure_c_helpers()
ENV_CONFIG = get_env_config()


DEFAULT_MAX_DEPTH = ENV_CONFIG.search_depth
DEFAULT_SEARCH_THREADS = max(1, ENV_CONFIG.search_threads)
c_helpers.set_max_search_depth(DEFAULT_MAX_DEPTH)

PIECE_SYMBOL_TO_INT = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": -1,
    "n": -2,
    "b": -3,
    "r": -4,
    "q": -5,
    "k": -6,
}


def _board_to_bitboard_state(board: chess.Board) -> c_helpers.BitboardState:
    pieces: list[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            pieces.append(0)
        else:
            pieces.append(PIECE_SYMBOL_TO_INT[piece.symbol()])

    castling = board.castling_xfen()
    ep_square = board.ep_square if board.ep_square is not None else -1
    return c_helpers.bitboard_from_components(
        pieces,
        board.turn == chess.WHITE,
        castling,
        ep_square,
        board.halfmove_clock,
        board.fullmove_number,
    )


def bitboard_from_fen(fen: str) -> c_helpers.BitboardState:
    """Convert a FEN string to a BitboardState."""
    return c_helpers.BitboardState(fen)


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


def alpha_beta(
    fen: str,
    depth: int,
    alpha: int = None,
    beta: int = None,
    maximizing_player: bool = None,
    tt=None,
    num_threads: int = DEFAULT_SEARCH_THREADS,
    killers=None,
    history=None,
    counters=None,
):
    """
    Call C++ alpha_beta function - full optimizations (TT, move ordering, etc.)
    """
    if alpha is None:
        alpha = c_helpers.MIN
    if beta is None:
        beta = c_helpers.MAX
    if maximizing_player is None:
        parts = fen.split()
        maximizing_player = len(parts) > 1 and parts[1] == "w"
    state = bitboard_from_fen(fen)
    return c_helpers.alpha_beta(
        state,
        depth,
        alpha,
        beta,
        maximizing_player,
        tt,
        num_threads,
        killers,
        history,
        counters,
    )


def alpha_beta_cuda(
    fen: str,
    depth: int,
    alpha: int = None,
    beta: int = None,
    maximizing_player: bool = None,
    tt=None,
    killers=None,
    history=None,
    counters=None,
):
    """
    Call C++ alpha_beta_cuda function - CUDA-accelerated search (falls back to optimized)
    """
    if alpha is None:
        alpha = c_helpers.MIN
    if beta is None:
        beta = c_helpers.MAX
    if maximizing_player is None:
        parts = fen.split()
        maximizing_player = len(parts) > 1 and parts[1] == "w"
    state = bitboard_from_fen(fen)
    return c_helpers.alpha_beta_cuda(
        state,
        depth,
        alpha,
        beta,
        maximizing_player,
        tt,
        killers,
        history,
        counters,
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
    num_threads: int = DEFAULT_SEARCH_THREADS,
    tt=None,
    killers=None,
    history=None,
    counters=None,
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
        counters: CounterMoveTable instance (creates new if None)

    Returns:
        Evaluation score
    """
    if counters is None:
        counters = c_helpers.CounterMoveTable()

    if use_cuda:
        # Try CUDA-accelerated search, fall back to CPU if it fails
        try:
            return alpha_beta_cuda(
                fen,
                depth,
                tt=tt,
                killers=killers,
                history=history,
                counters=counters,
            )
        except Exception as e:
            # CUDA failed, fall back to CPU
            print(f"CUDA search failed ({e}), falling back to CPU")
    # Use optimized CPU search with all features
    return alpha_beta(
        fen,
        depth,
        tt=tt,
        num_threads=num_threads,
        killers=killers,
        history=history,
        counters=counters,
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
    print("  Transposition table")
    print("  Move ordering (TT + MVV-LVA + promotions)")
    print("  Quiescence search")
    print("  Iterative deepening")
    if not use_cuda:
        print("  Parallel search")


if __name__ == "__main__":
    test_engine()
