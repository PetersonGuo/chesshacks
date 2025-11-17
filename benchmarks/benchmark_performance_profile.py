#!/usr/bin/env python3
"""
Performance profiling for chess engine
Compares: Normal (sequential), Multithreaded (parallel), and CUDA versions

Run from build directory: cd build && python ../benchmarks/benchmark_performance_profile.py
"""

import time

import c_helpers  # type: ignore

import benchmarks.conftest  # noqa: F401


def _state(fen: str) -> c_helpers.BitboardState:
    return c_helpers.BitboardState(fen)


def profile_search_methods():
    """Compare basic, optimized, and CUDA search performance."""
    print("=" * 80)
    print("CHESS ENGINE PERFORMANCE PROFILING")
    print("=" * 80)
    print()

    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    depth = 5

    def simple_evaluate(fen_str):
        """Simple material evaluation."""
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
        return sum(piece_values.get(c, 0) for c in fen_str.split()[0])

    print(f"Position: Italian Game (complex middlegame)")
    print(f"Depth: {depth}")
    print()

    # 1. Basic (no optimizations)
    print("1. BASIC SEARCH (no optimizations)")
    print("-" * 80)
    try:
        start = time.time()
        score_basic = c_helpers.alpha_beta(
            _state(fen), depth, c_helpers.MIN, c_helpers.MAX, True, simple_evaluate
        )
        time_basic = time.time() - start
        print(f"   Score: {score_basic:6d}")
        print(f"   Time:  {time_basic:.3f}s")
        print(f"   Nodes/s: {0:.0f} (no TT tracking)")
    except Exception as e:
        print(f"   Error: {e}")
        time_basic = None
        score_basic = None
    print()

    # 2. Optimized Sequential
    print("2. OPTIMIZED SEARCH (sequential, all optimizations)")
    print("-" * 80)
    tt_seq = c_helpers.TranspositionTable()
    killer = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    start = time.time()
    score_opt = c_helpers.alpha_beta(
        _state(fen),
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate,
        tt_seq,
        1,
        killer,
        history,
    )
    time_opt = time.time() - start
    nodes_opt = len(tt_seq)

    print(f"   Score: {score_opt:6d}")
    print(f"   Time:  {time_opt:.3f}s")
    print(f"   TT entries: {nodes_opt:6d}")
    print(f"   Nodes/s: {nodes_opt/time_opt:.0f}")

    if time_basic:
        speedup = time_basic / time_opt
        print(f"   Speedup vs Basic: {speedup:.1f}x")
    print()

    # 3. Optimized Multithreaded
    print("3. MULTITHREADED SEARCH (4 threads, all optimizations)")
    print("-" * 80)
    tt_par = c_helpers.TranspositionTable()
    killer_par = c_helpers.KillerMoves()
    history_par = c_helpers.HistoryTable()

    start = time.time()
    score_par = c_helpers.alpha_beta(
        _state(fen),
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate,
        tt_par,
        4,
        killer_par,
        history_par,
    )
    time_par = time.time() - start
    nodes_par = len(tt_par)

    print(f"   Score: {score_par:6d}")
    print(f"   Time:  {time_par:.3f}s")
    print(f"   TT entries: {nodes_par:6d}")
    print(f"   Nodes/s: {nodes_par/time_par:.0f}")

    if time_opt:
        speedup_par = time_opt / time_par
        print(f"   Speedup vs Sequential: {speedup_par:.2f}x")
        if speedup_par < 0.9:
            print(f"   ⚠️  Parallel slower (GIL contention or shallow depth)")
    print()

    # 4. CUDA
    print("4. CUDA SEARCH (GPU accelerated if available)")
    print("-" * 80)

    cuda_available = c_helpers.is_cuda_available()
    print(f"   CUDA Available: {cuda_available}")

    if cuda_available:
        cuda_info = c_helpers.get_cuda_info()
        print(f"   CUDA Info: {cuda_info}")

    tt_cuda = c_helpers.TranspositionTable()
    killer_cuda = c_helpers.KillerMoves()
    history_cuda = c_helpers.HistoryTable()

    start = time.time()
    score_cuda = c_helpers.alpha_beta_cuda(
        _state(fen),
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate,
        tt_cuda,
        killer_cuda,
        history_cuda,
    )
    time_cuda = time.time() - start
    nodes_cuda = len(tt_cuda)

    print(f"   Score: {score_cuda:6d}")
    print(f"   Time:  {time_cuda:.3f}s")
    print(f"   TT entries: {nodes_cuda:6d}")
    print(f"   Nodes/s: {nodes_cuda/time_cuda:.0f}")

    if cuda_available and time_opt:
        speedup_cuda = time_opt / time_cuda
        print(f"   Speedup vs Sequential: {speedup_cuda:.2f}x")
    elif not cuda_available:
        print(f"   Note: CUDA not available, using optimized CPU fallback")
    print()

    # Summary
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    results = []
    if time_basic:
        results.append(("Basic (no opts)", time_basic, 0, score_basic))
    results.append(("Optimized (1 thread)", time_opt, nodes_opt, score_opt))
    results.append(("Multithreaded (4 threads)", time_par, nodes_par, score_par))
    results.append(("CUDA (GPU)", time_cuda, nodes_cuda, score_cuda))

    # Sort by time (fastest first)
    results.sort(key=lambda x: x[1])

    print()
    print(f"{'Method':<30} {'Time':>10} {'Nodes':>10} {'Nodes/s':>12} {'Score':>8}")
    print("-" * 80)
    for method, t, nodes, score in results:
        nps = nodes / t if t > 0 and nodes > 0 else 0
        print(f"{method:<30} {t:>9.3f}s {nodes:>10d} {nps:>12.0f} {score:>8d}")

    print()
    print(f"Fastest method: {results[0][0]} ({results[0][1]:.3f}s)")

    if len(results) > 1:
        baseline = results[1][1]  # Use optimized sequential as baseline
        for method, t, _, _ in results:
            if method != "Optimized (1 thread)":
                speedup = baseline / t
                symbol = "🚀" if speedup > 1.1 else "⚠️" if speedup < 0.9 else "="
                print(f"{symbol} {method}: {speedup:.2f}x vs sequential")

    print()


def profile_evaluation_functions():
    """Profile different evaluation functions."""
    print()
    print("=" * 80)
    print("EVALUATION FUNCTION PROFILING")
    print("=" * 80)
    print()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    iterations = 100000

    # Material evaluation
    print("Testing evaluate (auto material/NNUE)...")
    state = _state(fen)
    start = time.time()
    for _ in range(iterations):
        c_helpers.evaluate(state)
    time_pst = time.time() - start

    print(f"   Iterations: {iterations:,}")
    print(f"   Total time: {time_pst:.3f}s")
    print(f"   Evaluations/s: {iterations/time_pst:,.0f}")
    print(f"   Time per eval: {time_pst/iterations*1000000:.2f} μs")
    print()


def profile_transposition_table():
    """Profile transposition table performance."""
    print("=" * 80)
    print("TRANSPOSITION TABLE PROFILING")
    print("=" * 80)
    print()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    print("TT Growth by Search Depth:")
    print("-" * 80)
    print(f"{'Depth':>6} {'TT Entries':>12} {'Time':>10} {'Nodes/s':>12}")
    print("-" * 80)

    for depth in range(2, 7):
        tt = c_helpers.TranspositionTable()
        start = time.time()
        c_helpers.get_best_move_uci(_state(fen), depth, c_helpers.evaluate, tt)
        elapsed = time.time() - start
        entries = len(tt)
        nps = entries / elapsed if elapsed > 0 else 0

        print(f"{depth:>6} {entries:>12,} {elapsed:>9.3f}s {nps:>12,.0f}")

    print()

    # TT Reuse test
    print("TT Reuse Test:")
    print("-" * 80)
    tt = c_helpers.TranspositionTable()

    print("First search (cold TT)...")
    start = time.time()
    c_helpers.get_best_move_uci(_state(fen), 5, c_helpers.evaluate, tt)
    time_cold = time.time() - start
    entries_cold = len(tt)

    print(f"   Time: {time_cold:.3f}s")
    print(f"   TT entries: {entries_cold:,}")

    print()
    print("Second search (warm TT, same position)...")
    start = time.time()
    c_helpers.get_best_move_uci(_state(fen), 5, c_helpers.evaluate, tt)
    time_warm = time.time() - start
    entries_warm = len(tt)

    print(f"   Time: {time_warm:.3f}s")
    print(f"   TT entries: {entries_warm:,}")
    print(f"   Speedup: {time_cold/time_warm:.1f}x")
    print()


def main():
    """Run all performance profiling tests."""
    try:
        profile_search_methods()
        profile_evaluation_functions()
        profile_transposition_table()

        print("=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user")
    except Exception as e:
        print(f"\n\nError during profiling: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
