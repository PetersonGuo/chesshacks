#!/usr/bin/env python3
"""
Helper script to profile the single-threaded vs multithreaded search.
Usage:
    python3 benchmarks/benchmark_profile_multithreading.py single
    python3 benchmarks/benchmark_profile_multithreading.py multi
"""

import sys
import time

import c_helpers  # type: ignore

import benchmarks.conftest  # noqa: F401


def _state(fen: str) -> c_helpers.BitboardState:
    return c_helpers.BitboardState(fen)


FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEPTH = 5
MODE = sys.argv[1] if len(sys.argv) > 1 else "single"
ITERATIONS = 3

if MODE not in {"single", "multi"}:
    raise SystemExit("Mode must be 'single' or 'multi'")

num_threads = 1 if MODE == "single" else 0  # 0 = auto (all cores)

print("=" * 80)
print(f"Profiling mode: {MODE} (num_threads={num_threads})")
print("=" * 80)

tt = c_helpers.TranspositionTable()
killers = c_helpers.KillerMoves()
history = c_helpers.HistoryTable()

start = time.perf_counter()
for i in range(ITERATIONS):
    score = c_helpers.alpha_beta(
        _state(FEN),
        DEPTH,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate,
        tt,
        num_threads,
        killers,
        history,
    )
    print(f"Iteration {i+1}: score={score}")

total = time.perf_counter() - start
print(f"Total time: {total:.4f}s")
print(f"Average per iteration: {total / ITERATIONS:.4f}s")
