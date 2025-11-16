"""
Chess Engine Utilities

This module contains helper functions and utilities for the chess engine.
These are not directly needed for the game connection but provide useful
functionality for testing and development.
"""

import sys
import os
import time
import chess

# Add build directory to path so c_helpers can be imported
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
if build_path not in sys.path:
    sys.path.insert(0, build_path)

# Add project root to path for CNN model (not train directory to avoid conflicts)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import c_helpers

# Cache original stderr reference (computed once)
_original_stderr = sys.__stderr__ if hasattr(sys, '__stderr__') else sys.stderr

# Global model instances (loaded lazily)
_cnn_model = None
_cnn_model_path = None
_cnn_evaluation_count = 0  # Track how many evaluations have been done

_nnue_model = None
_nnue_model_path = None
_nnue_evaluation_count = 0  # Track how many evaluations have been done

# Stockfish comparison
_stockfish_engine = None
_stockfish_depth = 10


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


def load_stockfish_engine():
    """Load Stockfish engine."""
    global _stockfish_engine
    if _stockfish_engine is not None:
        return _stockfish_engine
    try:
        import chess.engine
        _stockfish_engine = chess.engine.SimpleEngine.popen_uci('stockfish')
        return _stockfish_engine
    except Exception:
        return None


def evaluate_with_stockfish(fen: str) -> int:
    """Evaluate position with Stockfish."""
    global _stockfish_engine, _stockfish_depth
    if _stockfish_engine is None:
        load_stockfish_engine()
    if _stockfish_engine is None:
        return None
    try:
        board = chess.Board(fen)
        info = _stockfish_engine.analyse(board, chess.engine.Limit(depth=_stockfish_depth, time=2.0))
        score = info['score'].pov(chess.WHITE)
        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000
        return int(score.score())
    except Exception:
        return None


def enable_stockfish_comparison(enabled: bool = True, depth: int = 10):
    """Enable/disable Stockfish comparison."""
    global _stockfish_depth
    _stockfish_depth = depth
    if enabled:
        load_stockfish_engine()


# Normalization statistics from training data
# These are computed from the training dataset and used to denormalize model outputs
# IMPORTANT: These must match the stats used during training (after score clipping)
# With score clipping to [-1500, 1500], the actual training data stats are:
# Mean: 17.72, Std: 415.50 (computed from clipped train/cnn_model/data/train.jsonl)
# To convert normalized model output back to centipawns: centipawns = output * EVAL_STD + EVAL_MEAN
# NOTE: These default values will be overridden if the checkpoint contains normalization stats
EVAL_MEAN = 17.72  # Mean of training evaluations (in centipawns) after clipping
EVAL_STD = 415.50  # Std of training evaluations (in centipawns) after clipping

# Global variables to store model-specific normalization stats (loaded from checkpoint)
_cnn_eval_mean = EVAL_MEAN
_cnn_eval_std = EVAL_STD
_nnue_eval_mean = EVAL_MEAN
_nnue_eval_std = EVAL_STD


def load_cnn_model(model_path: str = None):
    """
    Load the CNN model from checkpoint.

    Args:
        model_path: Path to model checkpoint. If None, uses default best_model.pt

    Returns:
        Loaded CNN model
    """
    global _cnn_model, _cnn_model_path, _cnn_eval_mean, _cnn_eval_std
    
    if model_path is None:
        # Default to best_model.pt in train/cnn_model/checkpoints
        default_path = os.path.join(
            project_root, "train", "cnn_model", "checkpoints", "best_model.pt"
        )
        model_path = default_path
    
    # Only reload if path changed
    if _cnn_model is not None and _cnn_model_path == model_path:
        return _cnn_model
    
    try:
        import torch
        import chess
        from train.cnn_model.model import ChessCNNModel
        
        print(f"Loading CNN model from {model_path}...")
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            map_location = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            map_location = "mps"
        else:
            device = torch.device("cpu")
            map_location = "cpu"
        
        print(f"Using device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=map_location)
        
        # Get model config from checkpoint or use defaults
        config = checkpoint.get('config', {})
        model_config = {
            'input_channels': 12,
            'conv_channels': config.get('conv_channels', 64),
            'num_residual_blocks': config.get('num_residual_blocks', 4),
            'dense_hidden': config.get('dense_hidden', 256),
            'dropout': config.get('dropout', 0.3),
        }
        
        # Create model
        model = ChessCNNModel(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        _cnn_model = model
        _cnn_model_path = model_path

        # Load normalization stats from checkpoint if available
        if 'eval_mean' in checkpoint and 'eval_std' in checkpoint:
            _cnn_eval_mean = checkpoint['eval_mean']
            _cnn_eval_std = checkpoint['eval_std']
            print(f"[CNN] Model loaded successfully (device: {device})")
            if 'epoch' in checkpoint:
                print(f"[CNN]   Model trained for {checkpoint['epoch'] + 1} epochs")
            if 'val_loss' in checkpoint:
                print(f"[CNN]   Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"[CNN]   Normalization stats loaded from checkpoint:")
            print(f"[CNN]     mean={_cnn_eval_mean:.2f}, std={_cnn_eval_std:.2f}")
        else:
            # Use default stats and warn
            _cnn_eval_mean = EVAL_MEAN
            _cnn_eval_std = EVAL_STD
            print(f"[CNN] Model loaded successfully (device: {device})")
            if 'epoch' in checkpoint:
                print(f"[CNN]   Model trained for {checkpoint['epoch'] + 1} epochs")
            if 'val_loss' in checkpoint:
                print(f"[CNN]   Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"[CNN]   WARNING: Checkpoint missing normalization stats!")
            print(f"[CNN]   Using default values: mean={_cnn_eval_mean:.2f}, std={_cnn_eval_std:.2f}")
            print(f"[CNN]   This may cause incorrect evaluations if model was trained with different stats.")

        print(f"[CNN]   Ready for evaluation!")
        print(f"[CNN]   Evaluation counter initialized (will show progress every 10 evaluations)")

        return model
        
    except ImportError as e:
        print(f"[CNN] Warning: Could not load CNN model - missing dependency: {e}")
        print("[CNN] Falling back to nnue_evaluate")
        return None
    except Exception as e:
        print(f"[CNN] Warning: Could not load CNN model: {e}")
        print("[CNN] Falling back to nnue_evaluate")
        return None


def load_nnue_model(model_path: str = None):
    """
    Load the NNUE model from checkpoint.

    Args:
        model_path: Path to model checkpoint. If None, uses default best_model.pt

    Returns:
        Loaded NNUE model
    """
    global _nnue_model, _nnue_model_path, _nnue_eval_mean, _nnue_eval_std
    
    if model_path is None:
        # Default to best_model.pt in train/nnue_model/checkpoints
        default_path = os.path.join(
            project_root, "train", "nnue_model", "checkpoints", "best_model.pt"
        )
        model_path = default_path
    
    # Only reload if path changed
    if _nnue_model is not None and _nnue_model_path == model_path:
        return _nnue_model
    
    try:
        import torch
        import chess
        from train.nnue_model.model import ChessNNUEModel
        
        print(f"Loading NNUE model from {model_path}...")
        
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            map_location = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            map_location = "mps"
        else:
            device = torch.device("cpu")
            map_location = "cpu"
        
        print(f"Using device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=map_location)
        
        # Get model config from checkpoint or use defaults
        config = checkpoint.get('config', {})
        model_config = {
            'hidden_size': config.get('hidden_size', 256),
            'hidden2_size': config.get('hidden2_size', 32),
            'hidden3_size': config.get('hidden3_size', 32),
        }
        
        # Create model
        model = ChessNNUEModel(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        _nnue_model = model
        _nnue_model_path = model_path

        # Load normalization stats from checkpoint if available
        if 'eval_mean' in checkpoint and 'eval_std' in checkpoint:
            _nnue_eval_mean = checkpoint['eval_mean']
            _nnue_eval_std = checkpoint['eval_std']
            print(f"[NNUE] Model loaded successfully (device: {device})")
            if 'epoch' in checkpoint:
                print(f"[NNUE]   Model trained for {checkpoint['epoch'] + 1} epochs")
            if 'val_loss' in checkpoint:
                print(f"[NNUE]   Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"[NNUE]   Normalization stats loaded from checkpoint:")
            print(f"[NNUE]     mean={_nnue_eval_mean:.2f}, std={_nnue_eval_std:.2f}")
        else:
            # Use default stats and warn
            _nnue_eval_mean = EVAL_MEAN
            _nnue_eval_std = EVAL_STD
            print(f"[NNUE] Model loaded successfully (device: {device})")
            if 'epoch' in checkpoint:
                print(f"[NNUE]   Model trained for {checkpoint['epoch'] + 1} epochs")
            if 'val_loss' in checkpoint:
                print(f"[NNUE]   Validation loss: {checkpoint['val_loss']:.4f}")
            print(f"[NNUE]   WARNING: Checkpoint missing normalization stats!")
            print(f"[NNUE]   Using default values: mean={_nnue_eval_mean:.2f}, std={_nnue_eval_std:.2f}")
            print(f"[NNUE]   This may cause incorrect evaluations if model was trained with different stats.")

        print(f"[NNUE]   Ready for evaluation!")
        print(f"[NNUE]   Evaluation counter initialized (will show progress every 10 evaluations)")

        return model
        
    except ImportError as e:
        print(f"[NNUE] Warning: Could not load NNUE model - missing dependency: {e}")
        print("[NNUE] Falling back to nnue_evaluate")
        return None
    except Exception as e:
        print(f"[NNUE] Warning: Could not load NNUE model: {e}")
        print("[NNUE] Falling back to nnue_evaluate")
        return None


def sanitize_fen(fen: str) -> str:
    """
    Sanitize a FEN string to ensure it's valid for python-chess.
    Fixes negative fullmove numbers and halfmove clocks.
    
    Args:
        fen: FEN string (potentially invalid)
    
    Returns:
        Valid FEN string
    """
    parts = fen.split()
    if len(parts) < 6:
        # Invalid FEN, return as-is (will be caught by chess.Board)
        return fen
    
    # Ensure halfmove clock is non-negative
    try:
        halfmove = int(parts[4])
        if halfmove < 0:
            parts[4] = "0"
    except (ValueError, IndexError):
        parts[4] = "0"
    
    # Ensure fullmove number is at least 1
    try:
        fullmove = int(parts[5])
        if fullmove < 1:
            parts[5] = "1"
    except (ValueError, IndexError):
        parts[5] = "1"
    
    return " ".join(parts)


def nnue_evaluate_model(fen: str) -> int:
    """
    Evaluate a position using the NNUE model.

    Args:
        fen: FEN string of the position

    Returns:
        Evaluation score in centipawns (positive = white advantage)
    """
    global _nnue_model, _nnue_evaluation_count, _original_stderr, _nnue_eval_mean, _nnue_eval_std

    # Load model if not already loaded (only once)
    if _nnue_model is None:
        _nnue_model = load_nnue_model()
        if _nnue_model is None:
            return nnue_evaluate(fen)

    try:
        # Sanitize FEN to fix negative move numbers
        sanitized_fen = sanitize_fen(fen)
        board = chess.Board(sanitized_fen)

        # Evaluate using NNUE model
        normalized_score = _nnue_model.evaluate_board(board)

        # Denormalize to convert back to centipawns using model-specific stats
        centipawns = int(normalized_score * _nnue_eval_std + _nnue_eval_mean)
        
        # Track evaluation count
        _nnue_evaluation_count += 1
        
        # Minimal logging - only every 100th evaluation to reduce overhead
        if _nnue_evaluation_count % 100 == 0:
            _original_stderr.write(f"[NNUE] Evaluation #{_nnue_evaluation_count}...\n")
            _original_stderr.flush()
        
        return centipawns
        
    except Exception as e:
        # Only log errors, not every failure
        if _nnue_evaluation_count % 100 == 0:
            _original_stderr.write(f"[NNUE] Warning: Evaluation failed ({e})\n")
            _original_stderr.flush()
        return nnue_evaluate(fen)


def cnn_evaluate(fen: str) -> int:
    """
    Evaluate a position using the CNN model.

    Args:
        fen: FEN string of the position

    Returns:
        Evaluation score in centipawns (positive = white advantage)
    """
    global _cnn_model, _cnn_evaluation_count, _original_stderr, _cnn_eval_mean, _cnn_eval_std

    # Load model if not already loaded (only once)
    if _cnn_model is None:
        _cnn_model = load_cnn_model()
        if _cnn_model is None:
            return nnue_evaluate(fen)

    try:
        # Sanitize FEN to fix negative move numbers
        sanitized_fen = sanitize_fen(fen)
        board = chess.Board(sanitized_fen)

        # Evaluate using CNN model
        normalized_score = _cnn_model.evaluate_board(board)

        # Denormalize to convert back to centipawns using model-specific stats
        centipawns = int(normalized_score * _cnn_eval_std + _cnn_eval_mean)
        
        # Track evaluation count
        _cnn_evaluation_count += 1
        
        # Minimal logging - only every 100th evaluation to reduce overhead
        if _cnn_evaluation_count % 100 == 0:
            _original_stderr.write(f"[CNN] Evaluation #{_cnn_evaluation_count}...\n")
            _original_stderr.flush()
        
        return centipawns
        
    except Exception as e:
        # Only log errors, not every failure
        if _cnn_evaluation_count % 100 == 0:
            _original_stderr.write(f"[CNN] Warning: Evaluation failed ({e})\n")
            _original_stderr.flush()
        return nnue_evaluate(fen)


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
