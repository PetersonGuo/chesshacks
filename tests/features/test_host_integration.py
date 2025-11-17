"""
Host-level integration tests covering environment config, native loader
bootstrapping, and the ChessManager decorator utilities.
"""

from __future__ import annotations

import sys
import types

import chess
import pytest

from src import env_manager, native_loader
from src.utils.decorator import ChessManager

# ---------------------------------------------------------------------------
# Environment manager tests


@pytest.fixture(autouse=True)
def reset_env_cache(monkeypatch):
    """Clear env overrides and cached config before each test."""
    for key in (
        "CHESSHACKS_MAX_DEPTH",
        "CHESSHACKS_NUM_THREADS",
        "SERVE_PORT",
        "HF_TOKEN",
        "CHESSHACKS_ENABLE_CUDA",
        "CHESSHACKS_NNUE_MODEL",
        "TEST_ENV_INT",
    ):
        monkeypatch.delenv(key, raising=False)
    env_manager.get_env_config.cache_clear()
    yield
    env_manager.get_env_config.cache_clear()


def test_env_config_defaults():
    config = env_manager.get_env_config()
    assert config.search_depth == 4
    assert config.search_threads == 1
    assert config.serve_port == 5058
    assert config.hf_token is None
    assert config.cuda_enabled is False
    assert config.nnue_model_path == str(env_manager.DEFAULT_NNUE_PATH)


def test_env_config_overrides(monkeypatch):
    monkeypatch.setenv("CHESSHACKS_MAX_DEPTH", "12")
    monkeypatch.setenv("CHESSHACKS_NUM_THREADS", "8")
    monkeypatch.setenv("SERVE_PORT", "6060")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("CHESSHACKS_ENABLE_CUDA", "true")
    monkeypatch.setenv("CHESSHACKS_NNUE_MODEL", "/tmp/custom.pt")

    config = env_manager.get_env_config()
    assert config.search_depth == 12
    assert config.search_threads == 8
    assert config.serve_port == 6060
    assert config.hf_token == "secret"
    assert config.cuda_enabled is True
    assert config.nnue_model_path == "/tmp/custom.pt"


def test_env_bool_parsing(monkeypatch):
    monkeypatch.setenv("CHESSHACKS_ENABLE_CUDA", "0")
    assert env_manager.get_env_config().cuda_enabled is False
    monkeypatch.setenv("CHESSHACKS_ENABLE_CUDA", "YES")
    env_manager.get_env_config.cache_clear()
    assert env_manager.get_env_config().cuda_enabled is True


@pytest.mark.parametrize("value", ["-1", "0", "abc"])
def test_env_int_fallback(monkeypatch, value):
    monkeypatch.setenv("TEST_ENV_INT", value)
    assert env_manager._env_int("TEST_ENV_INT", 5) == 5


# ---------------------------------------------------------------------------
# Native loader tests


def _purge_paths(paths: set[str]):
    for entry in list(sys.path):
        if entry in paths:
            sys.path.remove(entry)


def test_ensure_sys_path_adds_project_and_build():
    target_paths = {str(native_loader.PROJECT_ROOT), str(native_loader.BUILD_DIR)}
    _purge_paths(target_paths)
    native_loader._ensure_sys_path()
    for expected in target_paths:
        assert expected in sys.path


def test_ensure_c_helpers_returns_existing_module(monkeypatch):
    dummy_module = types.SimpleNamespace(name="c_helpers")

    def fake_import(name):
        assert name == "c_helpers"
        return dummy_module

    monkeypatch.setattr(native_loader.importlib, "import_module", fake_import)

    native_loader.ensure_c_helpers.cache_clear()
    result = native_loader.ensure_c_helpers()
    assert result is dummy_module
    # Cached call should skip repeated imports
    assert native_loader.ensure_c_helpers() is dummy_module


def test_ensure_c_helpers_imports_once(monkeypatch):
    dummy_module = types.SimpleNamespace(name="c_helpers")

    calls = {"count": 0}

    def fake_import(name):
        calls["count"] += 1
        return dummy_module

    monkeypatch.setattr(native_loader.importlib, "import_module", fake_import)

    native_loader.ensure_c_helpers.cache_clear()
    assert native_loader.ensure_c_helpers() is dummy_module
    assert native_loader.ensure_c_helpers() is dummy_module
    assert calls["count"] == 1


# ---------------------------------------------------------------------------
# ChessManager decorator tests


def _reference_fen(moves: list[str]) -> str:
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board.fen()


def test_entrypoint_invocation_and_probability_logging():
    manager = ChessManager()
    expected_move = chess.Move.from_uci("e2e4")

    @manager.entrypoint
    def choose(ctx):
        ctx.logProbabilities({expected_move: 1.0})
        return expected_move

    move, probabilities, logs = manager.get_model_move()
    assert move == expected_move
    assert probabilities == {expected_move: 1.0}
    assert logs == ""


def test_set_context_parses_pgn_and_passes_timeleft():
    manager = ChessManager()
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    manager.set_context(pgn, timeleft=1500)

    expected_fen = _reference_fen(["e2e4", "e7e5", "g1f3", "b8c6"])
    returned_ctx: dict[str, object] = {}

    @manager.entrypoint
    def choose(ctx):
        returned_ctx["fen"] = ctx.board.fen()
        returned_ctx["time"] = ctx.timeLeft
        return chess.Move.from_uci("d2d4")

    manager.get_model_move()
    assert returned_ctx["fen"] == expected_fen
    assert returned_ctx["time"] == 1500


def test_call_reset_executes_handler_each_time():
    manager = ChessManager()
    calls = {"count": 0}

    @manager.reset
    def handle_reset(ctx):
        calls["count"] += 1

    manager.call_reset()
    manager.call_reset()
    assert calls["count"] == 2


def test_entrypoint_cannot_be_registered_twice():
    manager = ChessManager()

    @manager.entrypoint
    def first(ctx):
        return chess.Move.null()

    with pytest.raises(ValueError):

        @manager.entrypoint  # type: ignore[misc]
        def second(ctx):
            return chess.Move.null()

