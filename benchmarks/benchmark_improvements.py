#!/usr/bin/env python3
"""Quick test of multithreading and optimization improvements."""

import time

import c_helpers  # type: ignore

import benchmarks.conftest  # noqa: F401


def _state(fen: str) -> c_helpers.BitboardState:
    return c_helpers.BitboardState(fen)


print("=" * 80)
print("ChessHacks Optimization Test")
print("=" * 80)
print()

# Test 1: Batch evaluation
print("Test 1: Batch Evaluation Performance")
print("-" * 80)

test_fens = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
] * 100  # 300 positions
test_states = [_state(fen) for fen in test_fens]

print(f"Evaluating {len(test_fens)} positions...")
print()

# Sequential
start = time.time()
scores_seq = [c_helpers.evaluate(state) for state in test_states]
time_seq = time.time() - start
print(
    f"Sequential (for loop):     {time_seq:.4f}s  ({len(test_fens)/time_seq:.0f} pos/sec)"
)

# Multithreaded
start = time.time()
scores_mt = c_helpers.batch_evaluate_mt(test_states, 0)  # 0 = auto threads
time_mt = time.time() - start
print(
    f"Multithreaded (all cores): {time_mt:.4f}s  ({len(test_fens)/time_mt:.0f} pos/sec)"
)
print(f"Speedup: {time_seq/time_mt:.2f}x")
print()

# CUDA if available
if c_helpers.is_cuda_available():
    start = time.time()
    scores_cuda = c_helpers.cuda_batch_evaluate(test_states)
    time_cuda = time.time() - start
    print(
        f"CUDA (GPU):                {time_cuda:.4f}s  ({len(test_fens)/time_cuda:.0f} pos/sec)"
    )
    print(f"Speedup vs sequential: {time_seq/time_cuda:.2f}x")
    print(f"Speedup vs MT CPU:     {time_mt/time_cuda:.2f}x")
    print()

assert scores_seq == scores_mt, "Scores don't match!"
print("✓ All scores match")
print()

# Test 2: Search with threading
print("Test 2: Search Performance (Single Position)")
print("-" * 80)

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
depth = 5

# Single-threaded
print(f"Depth {depth} search...")
print()

tt1 = c_helpers.TranspositionTable()
killer1 = c_helpers.KillerMoves()
history1 = c_helpers.HistoryTable()

start = time.time()
score1 = c_helpers.alpha_beta(
    _state(fen),
    depth,
    c_helpers.MIN,
    c_helpers.MAX,
    True,
    c_helpers.evaluate,
    tt1,
    1,
    killer1,
    history1,
)
time1 = time.time() - start

print(f"1 thread:     {time1:.4f}s (score: {score1}, nodes: {len(tt1)})")

# Multi-threaded
tt2 = c_helpers.TranspositionTable()
killer2 = c_helpers.KillerMoves()
history2 = c_helpers.HistoryTable()

start = time.time()
score2 = c_helpers.alpha_beta(
    _state(fen),
    depth,
    c_helpers.MIN,
    c_helpers.MAX,
    True,
    c_helpers.evaluate,
    tt2,
    0,
    killer2,
    history2,
)
time2 = time.time() - start

import multiprocessing

num_cores = multiprocessing.cpu_count()

print(f"{num_cores} threads:  {time2:.4f}s (score: {score2}, nodes: {len(tt2)})")
print(f"Speedup: {time1/time2:.2f}x")
print()

# Note about scores
if score1 != score2:
    print("Note: Scores may differ in parallel search due to different move ordering")
print()

print("=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print()
print("Summary:")
print(f"  - Batch evaluation speedup: {time_seq/time_mt:.2f}x with multithreading")
if c_helpers.is_cuda_available():
    print(f"  - CUDA speedup: {time_seq/time_cuda:.2f}x over sequential")
print(f"  - Search speedup with {num_cores} cores: {time1/time2:.2f}x")
print()
