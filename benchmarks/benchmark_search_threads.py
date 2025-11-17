#!/usr/bin/env python3
"""
Benchmark multithreading improvements for ChessHacks.
Tests single-threaded vs multithreaded performance.
"""

import argparse
import multiprocessing
import threading
import time

import c_helpers  # type: ignore

import benchmarks.conftest  # noqa: F401
from env_manager import get_env_config

MIN_BENCH_DEPTH = 6
SCORE_TOLERANCE_CP = 50

ENV_CONFIG = get_env_config()
MAX_DEPTH = max(ENV_CONFIG.search_depth, MIN_BENCH_DEPTH)
c_helpers.set_max_search_depth(MAX_DEPTH)

PARALLEL_THREADS = max(
    16, min(multiprocessing.cpu_count(), ENV_CONFIG.search_threads or multiprocessing.cpu_count())
)


def _state(fen: str) -> c_helpers.BitboardState:
    return c_helpers.BitboardState(fen)


def benchmark_search(fen, depth, num_threads):
    """Run a single search benchmark with the requested thread count."""
    tt = c_helpers.TranspositionTable()
    killer = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    start = time.time()
    score = c_helpers.alpha_beta_builtin(
        _state(fen),
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        num_threads,
        killer,
        history,
        counters,
    )
    elapsed = time.time() - start

    return score, elapsed, len(tt)


def benchmark_batch_evaluation():
    """Benchmark the batch evaluation helper."""
    print("\n" + "=" * 80)
    print("Benchmark: Batch Evaluation Throughput")
    print("=" * 80)

    # Generate test positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
    ] * 100  # 500 positions

    print(f"\nEvaluating {len(test_fens)} positions...\n")

    test_states = [_state(fen) for fen in test_fens]

    # Sequential evaluation
    print("CPU sequential evaluation")
    start = time.time()
    scores_seq = [c_helpers.evaluate(state) for state in test_states]
    time_seq = time.time() - start
    print(f"   Time: {time_seq:.4f}s")
    print(f"   Rate: {len(test_fens)/time_seq:.0f} positions/sec")

    # Multithreaded batch evaluation (explicit threads)
    print(f"\nCPU multithreaded evaluation ({PARALLEL_THREADS} threads)")
    start = time.time()
    scores_mt = c_helpers.batch_evaluate_mt(test_states, PARALLEL_THREADS)
    time_mt = time.time() - start
    print(f"   Time: {time_mt:.4f}s")
    print(f"   Rate: {len(test_fens)/time_mt:.0f} positions/sec")
    print(f"   Speedup: {time_seq/time_mt:.2f}x")

    # Verify results match
    assert scores_seq == scores_mt, "MT scores don't match sequential!"
    print("\n✓ CPU sequential and multithreaded scores match exactly")


def main():
    parser = argparse.ArgumentParser(description="ChessHacks multithreading benchmark")
    parser.add_argument(
        "--depth",
        type=int,
        default=MAX_DEPTH,
        help=f"Search depth to benchmark (default: max of env depth and {MIN_BENCH_DEPTH})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=PARALLEL_THREADS,
        help="Parallel thread count (default: min(env threads, cpu count))",
    )
    args = parser.parse_args()

    bench_depth = max(args.depth, MIN_BENCH_DEPTH)
    parallel_threads = max(2, min(args.threads, multiprocessing.cpu_count()))

    c_helpers.set_max_search_depth(bench_depth)

    print("=" * 80)
    print("ChessHacks Multithreading Benchmark")
    print("=" * 80)

    # Test positions at different depths
    positions = [
        (
            "Starting position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "Complex middlegame",
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
        ),
        (
            "Tactical position",
            "r2qkb1r/ppp2ppp/2n5/3np1B1/2BPP1b1/2P2N2/PP3PPP/RN1Q1RK1 w kq - 0 9",
        ),
    ]

    depths = [bench_depth]
    thread_counts = [1, parallel_threads]

    print(f"\nSystem has {multiprocessing.cpu_count()} CPU cores available")
    print(f"Testing with {thread_counts[0]} thread(s) vs {thread_counts[1]} threads\n")
    print(f"Depths benchmarked: {depths}\n")

    for pos_name, fen in positions:
        print("\n" + "=" * 80)
        print(f"Position: {pos_name}")
        print("=" * 80)

        for depth in depths:
            print(f"\n--- Depth {depth} ---")

            results = {}
            for num_threads in thread_counts:
                label = (
                    "Sequential (1 thread)"
                    if num_threads == 1
                    else f"Parallel ({num_threads} threads)"
                )
                print(f"\n{label}:")

                score, elapsed, nodes = benchmark_search(fen, depth, num_threads)
                results[num_threads] = (score, elapsed, nodes)

                print(f"  Score: {score}")
                print(f"  Time: {elapsed:.4f}s")
                print(f"  Nodes: {nodes}")
                print(f"  NPS: {nodes/elapsed:.0f}")

            # Calculate speedup
            if len(results) == 2:
                seq_score, seq_time, _ = results[1]
                par_score, par_time, _ = results[parallel_threads]
                speedup = seq_time / par_time
                print(f"\n  Speedup (parallel vs sequential): {speedup:.2f}x")

                delta = abs(par_score - seq_score)
                if delta <= SCORE_TOLERANCE_CP:
                    print(f"  ✓ Scores within tolerance (Δ={delta} cp)")
                else:
                    print(
                        f"  ⚠ Score delta {delta} cp exceeds tolerance "
                        "(timing still recorded for reference)"
                    )

    # Batch evaluation benchmark
    benchmark_batch_evaluation()

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
