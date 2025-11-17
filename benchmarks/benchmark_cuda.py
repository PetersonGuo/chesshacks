#!/usr/bin/env python3
"""
Compare performance between CPU and CUDA implementations.
"""

import time

import c_helpers  # type: ignore

import benchmarks.conftest  # noqa: F401


def _state(fen: str) -> c_helpers.BitboardState:
    return c_helpers.BitboardState(fen)


def benchmark_search(name, search_func, fen, depth, num_runs=3):
    """Benchmark a search function."""
    print(f"\nBenchmarking {name} at depth {depth}...")
    print("-" * 60)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        score = search_func(fen, depth)
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s (score: {score})")

    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.4f}s")
    return avg_time


def main():
    print("=" * 60)
    print("ChessHacks CPU vs CUDA Performance Comparison")
    print("=" * 60)

    # Check CUDA availability
    cuda_available = c_helpers.is_cuda_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {c_helpers.get_cuda_info()}")

    # Test position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"\nTest Position: {fen}")

    # Create evaluation function
    def evaluate(fen_str):
        return c_helpers.evaluate(_state(fen_str))

    # Create search data structures
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    # Test different depths
    depths = [3, 4, 5]

    cpu_times = []
    cuda_times = []

    for depth in depths:
        print(f"\n{'=' * 60}")
        print(f"Depth {depth}")
        print("=" * 60)

        # CPU version
        tt.clear()
        killers.clear()
        history.clear()

        def cpu_search(fen, d):
            return c_helpers.alpha_beta(
                _state(fen),
                d,
                c_helpers.MIN,
                c_helpers.MAX,
                True,
                evaluate,
                tt,
                0,
                killers,
                history,
            )

        cpu_time = benchmark_search("CPU (Optimized)", cpu_search, fen, depth)
        cpu_times.append(cpu_time)

        # CUDA version (if available)
        if cuda_available:
            tt.clear()
            killers.clear()
            history.clear()

            def cuda_search(fen, d):
                return c_helpers.alpha_beta_cuda(
                    _state(fen),
                    d,
                    c_helpers.MIN,
                    c_helpers.MAX,
                    True,
                    evaluate,
                    tt,
                    killers,
                    history,
                )

            cuda_time = benchmark_search("CUDA (GPU)", cuda_search, fen, depth)
            cuda_times.append(cuda_time)

            # Calculate speedup
            speedup = cpu_time / cuda_time
            print(f"\n  Speedup: {speedup:.2f}x")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)

    print("\nAverage times by depth:")
    print(f"{'Depth':<10} {'CPU (s)':<15} {'CUDA (s)':<15} {'Speedup':<10}")
    print("-" * 60)

    for i, depth in enumerate(depths):
        cpu_time = cpu_times[i]
        if cuda_available and i < len(cuda_times):
            cuda_time = cuda_times[i]
            speedup = cpu_time / cuda_time
            print(f"{depth:<10} {cpu_time:<15.4f} {cuda_time:<15.4f} {speedup:<10.2f}x")
        else:
            print(f"{depth:<10} {cpu_time:<15.4f} {'N/A':<15} {'N/A':<10}")

    if cuda_available and cuda_times:
        avg_speedup = sum(
            cpu_times[i] / cuda_times[i] for i in range(len(cuda_times))
        ) / len(cuda_times)
        print(f"\nOverall average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
