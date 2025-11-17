"""
CUDA feature tests consolidated into one pytest module.

Each test first validates the CPU-side detection APIs. CUDA-specific operations
are skipped gracefully when the extension is built without GPU support.
"""

from __future__ import annotations

import subprocess

import c_helpers
import pytest


@pytest.fixture(scope="session")
def cuda_status():
    available = getattr(c_helpers, "is_cuda_available", lambda: False)()
    info = getattr(c_helpers, "get_cuda_info", lambda: "Unavailable")
    return bool(available), str(info)


def test_cuda_detection_reports_status(cuda_status):
    available, info = cuda_status
    assert isinstance(available, bool)
    assert isinstance(info, str)


def test_cuda_cli_detection():
    """nvidia-smi is optional, but when present it should return without error."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=2, check=False
        )
    except FileNotFoundError:
        pytest.skip("nvidia-smi not installed")
    except subprocess.TimeoutExpired:
        pytest.skip("nvidia-smi timed out")
    else:
        assert result.returncode in (0, 3)  # 3 happens on WSL without GPU


def _require_cuda(cuda_status, attr: str):
    available, _ = cuda_status
    if not available or not hasattr(c_helpers, attr):
        pytest.skip("CUDA support not available in this build")


def test_cuda_batch_evaluate(cuda_status, make_state):
    _require_cuda(cuda_status, "cuda_batch_evaluate")
    boards = [make_state(fen) for fen in [STARTING_FEN, ADVANTAGE_FEN]]
    scores = c_helpers.cuda_batch_evaluate(boards)
    assert len(scores) == len(boards)


def test_cuda_batch_piece_counts(cuda_status, make_state):
    _require_cuda(cuda_status, "cuda_batch_count_pieces")
    boards = [make_state(STARTING_FEN)]
    counts = c_helpers.cuda_batch_count_pieces(boards)
    assert len(counts) == len(boards)
    assert len(counts[0]) == 12


def test_cuda_batch_hashes(cuda_status, make_state):
    _require_cuda(cuda_status, "cuda_batch_hash_positions")
    boards = [make_state(STARTING_FEN), make_state(ADVANTAGE_FEN)]
    hashes = c_helpers.cuda_batch_hash_positions(boards)
    assert len(hashes) == len(boards)
    assert hashes[0] != hashes[1]


def test_alpha_beta_cuda_falls_back_or_runs(cuda_status, make_state):
    if not hasattr(c_helpers, "alpha_beta_cuda"):
        pytest.skip("Extension built without alpha_beta_cuda symbol")

    state = make_state(STARTING_FEN)
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    score = c_helpers.alpha_beta_cuda(
        state,
        3,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate,
        tt,
        killers,
        history,
    )
    assert isinstance(score, int)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ADVANTAGE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
