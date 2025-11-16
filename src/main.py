"""
Chess Engine Main - Game Connection Interface

Provides the entrypoint/reset hooks required by the chess_manager decorator.
All internal state stays in Virgo bitboards; FEN is used only at the API edges.
"""

import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

import chess
from chess import Move

if __package__ in (None, ""):
    # Allow running as `python src/main.py`
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from env_manager import get_env_config
    from native_loader import ensure_c_helpers
    from utils import GameContext, chess_manager
    import engine  # type: ignore
else:
    from . import engine
    from .env_manager import get_env_config
    from .native_loader import ensure_c_helpers
    from .utils import GameContext, chess_manager

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_root = os.path.join(project_root, "train")
ENV_CONFIG = get_env_config()


# Optional NNUE model (falls back silently if file missing)
NNUE_MODEL_PATH = os.path.abspath(ENV_CONFIG.nnue_model_path)
NNUE_TORCHSCRIPT_PATH: Optional[str] = None
NNUE_LOADED = False


def _is_torchscript_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = archive.namelist()
            has_data = any(name.endswith("data.pkl") for name in entries)
            has_code = any("__torch__" in name for name in entries)
            return has_data and has_code
    except zipfile.BadZipFile:
        return False


def _convert_checkpoint_to_torchscript(source_path: Path) -> Optional[str]:
    target_path = source_path.with_suffix(".torchscript.pt")
    try:
        import torch
    except ImportError as exc:
        print(
            f"PyTorch not available; cannot convert NNUE checkpoint to TorchScript. ({exc})"
        )
        return _convert_checkpoint_via_subprocess(source_path, target_path)

    sys_path_updated = False
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
        sys_path_updated = True

    try:
        from train.nnue_model.model import ChessNNUEModel  # type: ignore
    except Exception as exc:
        print(f"Unable to import NNUE model definition: {exc}")
        if sys_path_updated and train_root in sys.path:
            sys.path.remove(train_root)
        return None

    try:
        ckpt = torch.load(str(source_path), map_location="cpu")
    except Exception as exc:
        print(f"Failed to load NNUE checkpoint {source_path}: {exc}")
        if sys_path_updated and train_root in sys.path:
            sys.path.remove(train_root)
        return None

    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = ChessNNUEModel()
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    try:
        scripted = torch.jit.script(model)
    except Exception as exc:
        print(f"Failed to convert NNUE checkpoint to TorchScript: {exc}")
        if sys_path_updated and train_root in sys.path:
            sys.path.remove(train_root)
        return None

    try:
        scripted.save(str(target_path))
        print(f"Exported TorchScript NNUE model to {target_path}")
        return str(target_path)
    except Exception as exc:
        print(f"Failed to save TorchScript NNUE model: {exc}")
        return _convert_checkpoint_via_subprocess(source_path, target_path)
    finally:
        if sys_path_updated and train_root in sys.path:
            sys.path.remove(train_root)


def _convert_checkpoint_via_subprocess(
    source_path: Path, target_path: Path
) -> Optional[str]:
    export_script = Path(project_root) / "train" / "nnue_model" / "export_torchscript.py"
    if not export_script.exists():
        print(
            f"TorchScript helper script missing at {export_script}; cannot convert NNUE model."
        )
        return None

    cmd = [
        sys.executable,
        str(export_script),
        str(source_path),
        "--output",
        str(target_path),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as exc:
        print(f"TorchScript export helper failed ({exc})")
        return None

    if _is_torchscript_artifact(target_path):
        print(f"Exported TorchScript NNUE model via helper: {target_path}")
        return str(target_path)

    print("TorchScript export helper did not produce a valid artifact.")
    return None


def _resolve_torchscript_model_path(model_path: str) -> Optional[str]:
    candidate = Path(model_path)
    if not candidate.exists():
        print(f"NNUE model file not found at {candidate}")
        return None

    if _is_torchscript_artifact(candidate):
        return str(candidate)

    converted_path = candidate.with_suffix(".torchscript.pt")
    if _is_torchscript_artifact(converted_path):
        return str(converted_path)

    return _convert_checkpoint_to_torchscript(candidate)


def load_nnue_model() -> bool:
    global NNUE_LOADED, NNUE_TORCHSCRIPT_PATH

    if NNUE_LOADED:
        return True

    if NNUE_TORCHSCRIPT_PATH is None:
        NNUE_TORCHSCRIPT_PATH = _resolve_torchscript_model_path(NNUE_MODEL_PATH)

    if NNUE_TORCHSCRIPT_PATH is None:
        return False

    try:
        NNUE_LOADED = c_helpers.init_nnue(NNUE_TORCHSCRIPT_PATH)
    except Exception as exc:
        print(f"NNUE load failed ({exc})")
        NNUE_LOADED = False

    return NNUE_LOADED


NNUE_TORCHSCRIPT_PATH = _resolve_torchscript_model_path(NNUE_MODEL_PATH)
c_helpers = ensure_c_helpers()

# Global configuration
SEARCH_DEPTH = ENV_CONFIG.search_depth
c_helpers.set_max_search_depth(SEARCH_DEPTH)
NUM_THREADS = 0  # 0 = auto-detect CPU cores
PROBABILITY_TEMPERATURE = 200.0

# Persistent native resources
transposition_table = c_helpers.TranspositionTable()
killer_moves = c_helpers.KillerMoves()
history_table = c_helpers.HistoryTable()
counter_move_table = c_helpers.CounterMoveTable()

# Capability detection
USE_CUDA = engine.has_cuda()

load_nnue_model()


def _active_eval():
    """Select the fastest available native evaluator."""
    if NNUE_LOADED:
        return c_helpers.evaluate
    return c_helpers.evaluate_with_pst

print("Chess Engine initialized:")
print(f"  - CUDA available: {USE_CUDA}")
if USE_CUDA:
    print(f"  - CUDA info: {engine.get_cuda_status()}")
print(f"  - NNUE loaded: {NNUE_LOADED}")
if NNUE_LOADED and NNUE_TORCHSCRIPT_PATH:
    print(f"  - NNUE model: {NNUE_TORCHSCRIPT_PATH}")
elif NNUE_TORCHSCRIPT_PATH:
    print(f"  - NNUE TorchScript candidate: {NNUE_TORCHSCRIPT_PATH}")
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


def _build_probability_distribution(board: chess.Board) -> Dict[Move, float]:
    """Use multi-PV search (bitboard-native) to estimate move probabilities."""
    board_snapshot = board.copy()
    legal_moves = set(board_snapshot.generate_legal_moves())
    if not legal_moves:
        raise ValueError("No legal moves available for probability distribution")

    max_lines = min(5, len(legal_moves))
    score_map: Dict[Move, float] = {}

    try:
        bitboard_state = engine._board_to_bitboard_state(board_snapshot)
        pv_lines = c_helpers.multi_pv_search_state(
            bitboard_state,
            SEARCH_DEPTH,
            max_lines,
            _active_eval(),
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
    except Exception as exc:
        print(f"multi_pv_search failed ({exc}); falling back to heuristics")
        return _fallback_probability_distribution(board_snapshot)

    if not score_map:
        return _fallback_probability_distribution(board_snapshot)
    return score_map


def _search_best_move(bitboard_state: c_helpers.BitboardState) -> str:
    """Search for the best move using the fastest native evaluation path."""
    return c_helpers.get_best_move_uci_state(
        bitboard_state,
        SEARCH_DEPTH,
        _active_eval(),
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

    print(f"Engine selected: {best_move_uci}")

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    try:
        best_move = Move.from_uci(best_move_uci)
    except ValueError:
        print(f"Warning: Failed to parse UCI move '{best_move_uci}'")
        best_move = legal_moves[0]

    score_map = _build_probability_distribution(ctx.board)
    if best_move not in score_map:
        score_map[best_move] = 0.0

    if not score_map:
        score_map = {best_move: 1.0}

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
