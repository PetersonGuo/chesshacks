#!/usr/bin/env python3
"""
Benchmark harness for native (C++) engine functions exposed via c_helpers.

This script times the core search/evaluation entrypoints so we can monitor
regressions in the C++ layer without relying on the FastAPI serve path.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Dict, List, Tuple

import c_helpers  # type: ignore
import chess

import benchmarks.conftest  # noqa: F401
import engine
from env_manager import get_env_config

ENV_CONFIG = get_env_config()
DEFAULT_DEPTH = ENV_CONFIG.search_depth
DEFAULT_THREADS = max(1, ENV_CONFIG.search_threads)

try:
    import main as main_module  # type: ignore

    main_module.load_nnue_model()
    ACTIVE_EVAL_FN = c_helpers.evaluate
    NNUE_ACTIVE = True
except Exception:  # pragma: no cover - optional dependency
    ACTIVE_EVAL_FN = c_helpers.evaluate
    NNUE_ACTIVE = False


POSITIONS: List[Tuple[str, str]] = [
    ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    (
        "Equal middlegame",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
    ),
    (
        "Tactical slugfest",
        "r2qkb1r/ppp2ppp/2n5/3np1B1/2BPP1b1/2P2N2/PP3PPP/RN1Q1RK1 w kq - 0 9",
    ),
]


def _make_resources():
    return (
        c_helpers.TranspositionTable(),
        c_helpers.KillerMoves(),
        c_helpers.HistoryTable(),
        c_helpers.CounterMoveTable(),
    )


def _board_state(fen: str) -> c_helpers.BitboardState:
    board = chess.Board(fen)
    return engine._board_to_bitboard_state(board)  # type: ignore[attr-defined]


def _benchmark(
    name: str, runs: int, func: Callable[[], None]
) -> Dict[str, float | List[float]]:
    durations: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    avg = statistics.mean(durations)
    return {
        "name": name,
        "runs": runs,
        "avg_ms": avg * 1000,
        "min_ms": min(durations) * 1000,
        "max_ms": max(durations) * 1000,
        "samples": durations,
    }


def bench_get_best_move(fen: str, depth: int, threads: int, runs: int):
    def call():
        state = _board_state(fen)
        tt, killers, history, counters = _make_resources()
        c_helpers.get_best_move_uci_state(
            state,
            depth,
            ACTIVE_EVAL_FN,
            tt,
            threads,
            killers,
            history,
            counters,
        )

    return _benchmark(f"get_best_move_uci_state depth={depth}", runs, call)


def bench_multi_pv(fen: str, depth: int, threads: int, runs: int):
    def call():
        state = _board_state(fen)
        tt, killers, history, counters = _make_resources()
        c_helpers.multi_pv_search_state(
            state,
            depth,
            min(5, len(list(chess.Board(fen).generate_legal_moves()))),
            ACTIVE_EVAL_FN,
            tt,
            threads,
            killers,
            history,
            counters,
        )

    return _benchmark(f"multi_pv_search_state depth={depth}", runs, call)


def bench_batch_eval(fens: List[str], threads: int, runs: int):
    def call():
        states = [_board_state(fen) for fen in fens]
        c_helpers.batch_evaluate_mt(states, threads)

    return _benchmark(
        f"batch_evaluate_mt positions={len(fens)} threads={threads}", runs, call
    )


def bench_single_eval(fen: str, runs: int):
    def call():
        ACTIVE_EVAL_FN(_board_state(fen))

    mode = "NNUE" if NNUE_ACTIVE else "Material"
    return _benchmark(f"{mode} evaluate", runs, call)


def print_result(result: Dict[str, float | List[float]]):
    name = result["name"]
    avg = result["avg_ms"]
    min_ms = result["min_ms"]
    max_ms = result["max_ms"]
    runs = result["runs"]
    print(
        f"{name:40s} | runs={runs:2d} | avg={avg:8.2f} ms | min={min_ms:8.2f} | max={max_ms:8.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark native C++ engine calls")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Search depth")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Threads for non-batch search (1 = sequential)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of repetitions per benchmark"
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=len(POSITIONS),
        help="Number of predefined FENs to benchmark (prefix)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of FENs for batch evaluate benchmark",
    )
    args = parser.parse_args()

    positions = POSITIONS[: args.positions]
    if not positions:
        print("No positions selected; exiting.")
        return

    print("=== C++ Function Benchmarks ===")
    print(f"Depth: {args.depth}, Threads: {args.threads}, Runs: {args.runs}")
    print(f"Evaluation mode: {'NNUE' if NNUE_ACTIVE else 'Material'}")
    print("")

    for label, fen in positions:
        print(f"--- {label} ---")
        print_result(bench_single_eval(fen, args.runs))
        print_result(bench_get_best_move(fen, args.depth, args.threads, args.runs))
        print_result(bench_multi_pv(fen, args.depth, args.threads, args.runs))
        print("")

    repeated_fens = [fen for _, fen in positions]
    while len(repeated_fens) < args.batch_size:
        repeated_fens.extend(fen for _, fen in positions)
    repeated_fens = repeated_fens[: args.batch_size]
    print_result(bench_batch_eval(repeated_fens, args.threads, args.runs))


if __name__ == "__main__":
    main()
