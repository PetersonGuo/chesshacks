"""
Chess Engine Main - Game Connection Interface

This module provides the entrypoint and reset functions required by the
chess_manager decorator system for connecting to the game server.
"""

from .utils import chess_manager, GameContext
from chess import Move
import sys
import os
import subprocess

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
                text=True
            )
            print(result.stdout)
            
            # Try import again
            import c_helpers
            print("✓ Build successful - module imported")
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            
            error_output = e.stderr + "\n" + (e.stdout or "")
            
            if "cmake: command not found" in error_output:
                print("\n" + "=" * 80)
                print("ERROR: cmake is not installed or not in PATH")
                print("=" * 80)
                print("To fix this, install cmake:")
                print("  macOS: brew install cmake")
                print("  Linux: sudo apt-get install cmake (or use your package manager)")
                print("  Or download from: https://cmake.org/download/")
                print("\nAfter installing cmake, the build should work automatically.")
                print("=" * 80)
            elif ("find_package(Python" in error_output or "Python Development" in error_output or 
                  "Python3_EXECUTABLE" in error_output or "CMakeLists.txt:48" in error_output or
                  "Could not find a package configuration file" in error_output and "Python" in error_output):
                print("\n" + "=" * 80)
                print("ERROR: Python development headers not found")
                print("=" * 80)
                print("CMake needs Python development headers to build the C++ module.")
                print("\nTo fix this:")
                print("  macOS: Install Python with development headers:")
                print("    brew install python@3.11  (or your Python version)")
                print("    Or ensure python3-dev/python3-devel is installed")
                print("\n  Linux: Install python3-dev:")
                print("    sudo apt-get install python3-dev  (Debian/Ubuntu)")
                print("    sudo yum install python3-devel    (RHEL/CentOS)")
                print("\n  Verify Python is found:")
                print("    python3 --version")
                print("    python3-config --includes  (should show include paths)")
                print("=" * 80)
            elif "find_package(nanobind" in error_output or "nanobind" in error_output.lower():
                print("\n" + "=" * 80)
                print("ERROR: nanobind package not found")
                print("=" * 80)
                print("The build requires nanobind for Python bindings.")
                print("\nTo fix this, install nanobind:")
                print("    pip install nanobind")
                print("\nThen try building again.")
                print("=" * 80)
            raise
        except ModuleNotFoundError:
            print("Build completed but module still not found.")
            print("Please run: ./build.sh manually to see detailed build output")
            raise
    else:
        print(f"Build script not found at: {build_script}")
        print("Please run: ./build.sh from project root")
        raise

# Import engine utilities
from . import engine

# Debug: Verify entrypoint registration
import sys
sys.stderr.write("[MAIN] main.py loaded, registering make_move entrypoint...\n")
sys.stderr.flush()

# Global configuration
SEARCH_DEPTH = 3  # Depth 5 averages ~200-1200ms, safe for 1-minute games
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

# Load CNN model at startup
print("=" * 60)
print("Loading CNN evaluation model...")
print("=" * 60)
cnn_model = engine.load_cnn_model()
nnue_model = None  # Initialize for fallback
if cnn_model is not None:
    print("[CNN] ✓ CNN model loaded - using CNN for board evaluation")
    print("[CNN] All position evaluations will use the CNN model")
else:
    print("[CNN] ✗ CNN model not available - trying NNUE fallback...")
    nnue_model = engine.load_nnue_model()
    if nnue_model is not None:
        print("[NNUE] ✓ NNUE model loaded - using NNUE for board evaluation")
    else:
        print("[NNUE] ✗ NNUE model not available - falling back to material evaluation")
print("=" * 60)

# Stockfish comparison is disabled during search to avoid conflicts
# We'll only compare after moves are made
print("=" * 60)
print("Stockfish comparison setup...")
print("=" * 60)
engine.enable_stockfish_comparison(enabled=False, depth=10)  # Disabled during search
print("[Stockfish] Comparison disabled during search (will compare after each move)")
print("=" * 60)

# Debug: Confirm we're about to register entrypoint
import sys
sys.stderr.write("[MAIN] About to register @chess_manager.entrypoint decorator...\n")
sys.stderr.flush()

@chess_manager.entrypoint
def make_move(ctx: GameContext) -> Move:
    """
    Main entrypoint - called every time the engine needs to make a move.
    Returns a python-chess Move object that is a legal move for the current position.
    """
    # Get original stderr before any redirects (from wrapper's context)
    import sys
    # Try to get original stderr - the wrapper should have set this up
    # If we're in a redirect context, sys.__stderr__ should still point to real stderr
    original_stderr = sys.__stderr__ if hasattr(sys, '__stderr__') else sys.stderr
    
    original_stderr.write(f"[MAKE_MOVE] ===== make_move() CALLED =====\n")
    original_stderr.write(f"[MAKE_MOVE] Thinking... (depth={SEARCH_DEPTH})\n")
    
    # Determine which evaluation function to use
    if cnn_model is not None:
        eval_func = engine.cnn_evaluate
        eval_name = "CNN"
    elif nnue_model is not None:
        eval_func = engine.nnue_evaluate_model
        eval_name = "NNUE"
    else:
        eval_func = engine.nnue_evaluate
        eval_name = "Material"
    
    original_stderr.write(f"[MAKE_MOVE] [{eval_name}] ===== Using {eval_name} model for position evaluation =====\n")
    original_stderr.flush()
    
    print(f"Thinking... (depth={SEARCH_DEPTH})")
    print(f"[{eval_name}] ===== Using {eval_name} model for position evaluation =====")

    # Get current position as FEN
    fen = ctx.board.fen()
    original_stderr.write(f"[MAKE_MOVE] Position FEN: {fen}\n")
    original_stderr.write(f"[MAKE_MOVE] Starting search ({eval_name} will evaluate positions during search)...\n")
    original_stderr.write(f"[MAKE_MOVE] About to call c_helpers.get_best_move_uci()...\n")
    original_stderr.flush()
    
    print(f"[{eval_name}] Position FEN: {fen}")
    print(f"[{eval_name}] Starting search ({eval_name} will evaluate positions during search)...")

    # Use C++ engine to find best move with NNUE/CNN evaluation
    # Try with CUDA if available, fall back to CPU on error
    try:
        original_stderr.write(f"[MAKE_MOVE] Calling get_best_move_uci NOW...\n")
        original_stderr.flush()
        best_move_uci = c_helpers.get_best_move_uci(
            fen,
            SEARCH_DEPTH,
            eval_func,  # Use NNUE/CNN evaluation
            transposition_table,
            NUM_THREADS,
            killer_moves,
            history_table,
            c_helpers.CounterMoveTable(),  # Create counter move table
        )
        original_stderr.write(f"[MAKE_MOVE] get_best_move_uci() returned: {best_move_uci}\n")
        original_stderr.flush()
    except Exception as e:
        # If CUDA or any C++ error occurs, log and retry with fallback parameters
        original_stderr.write(f"[MAKE_MOVE] Exception in get_best_move_uci: {e}\n")
        original_stderr.write(f"[MAKE_MOVE] Retrying with fallback parameters...\n")
        original_stderr.flush()
        print(f"Engine error ({e}), retrying with fallback...")
        best_move_uci = c_helpers.get_best_move_uci(
            fen,
            SEARCH_DEPTH,
            engine.cnn_evaluate,  # Use CNN evaluation instead of nnue_evaluate
            transposition_table,
            0,  # Single thread as fallback
            killer_moves,
            history_table,
            c_helpers.CounterMoveTable(),
        )
        original_stderr.write(f"[MAKE_MOVE] Fallback get_best_move_uci() returned: {best_move_uci}\n")
        original_stderr.flush()

    print(f"Engine selected: {best_move_uci}")
    # Import to get evaluation count
    from src.engine import _cnn_evaluation_count
    original_stderr.write(f"[MAKE_MOVE] Engine selected: {best_move_uci}\n")
    original_stderr.write(f"[MAKE_MOVE] Total CNN evaluations: {_cnn_evaluation_count}\n")
    original_stderr.write(f"[MAKE_MOVE] ===== make_move() COMPLETE =====\n")
    original_stderr.flush()
    
    print(f"[CNN] ===== Move selection complete =====")
    print(f"[CNN] Total CNN evaluations performed: {_cnn_evaluation_count}")

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

    # Compare CNN evaluation with Stockfish after the move
    test_board = ctx.board.copy()
    test_board.push(best_move)
    position_after_move = test_board.fen()
    
    # Get CNN evaluation (comparison is already disabled during search)
    if cnn_model is not None:
        cnn_eval = engine.cnn_evaluate(position_after_move)
    elif nnue_model is not None:
        cnn_eval = engine.nnue_evaluate_model(position_after_move)
    else:
        cnn_eval = engine.nnue_evaluate(position_after_move)
    
    # Get Stockfish evaluation (only after move, not during search)
    try:
        stockfish_eval = engine.evaluate_with_stockfish(position_after_move)
    except Exception:
        stockfish_eval = None
    
    # Show comparison
    if stockfish_eval is not None:
        error = abs(cnn_eval - stockfish_eval)
        print(f"\n[Comparison] CNN: {cnn_eval} cp | Stockfish: {stockfish_eval} cp | Error: {error} cp")
    else:
        print(f"\n[Comparison] CNN: {cnn_eval} cp | Stockfish: unavailable")

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
