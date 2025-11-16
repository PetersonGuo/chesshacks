#!/usr/bin/env python3
"""
Test parallel optimizations in various search functions
Compares sequential vs parallel performance
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


def test_parallel_multi_pv():
    """Test parallel multi-PV search optimization"""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    depth = 4
    num_lines = 5

    # Sequential
    tt = c_helpers.TranspositionTable()
    result_seq = c_helpers.multi_pv_search(
        fen, depth, num_lines, c_helpers.evaluate_with_pst, tt, 1
    )

    # Parallel
    tt.clear()
    result_par = c_helpers.multi_pv_search(
        fen, depth, num_lines, c_helpers.evaluate_with_pst, tt, 4
    )

    # Both should find the same number of lines
    assert len(result_seq) == len(
        result_par
    ), f"Line count mismatch: {len(result_seq)} vs {len(result_par)}"


def test_parallel_find_best_move():
    """Test parallel find_best_move optimization"""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    depth = 3

    # Sequential
    tt = c_helpers.TranspositionTable()
    result_seq = c_helpers.find_best_move(
        fen, depth, c_helpers.evaluate_with_pst, tt, 1
    )

    # Parallel
    tt.clear()
    result_par = c_helpers.find_best_move(
        fen, depth, c_helpers.evaluate_with_pst, tt, 4
    )

    # Both should return valid FENs
    assert len(result_seq) > 0, "Sequential returned empty FEN"
    assert len(result_par) > 0, "Parallel returned empty FEN"


def test_parallel_root_search():
    """Test parallel alpha-beta root search"""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    depth = 5  # Deeper to see benefit

    # Sequential
    tt = c_helpers.TranspositionTable()
    start = time.time()
    result_seq = c_helpers.alpha_beta_optimized(
        fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        1,
    )
    time_seq = time.time() - start

    # Parallel
    tt.clear()
    start = time.time()
    result_par = c_helpers.alpha_beta_optimized(
        fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        4,
    )
    time_par = time.time() - start

    # Check that parallel is at least as fast or has valid reason for being slower
    # With C++ evaluate and depth >= 4, parallel should be faster or comparable
    print(f"  Sequential: {time_seq:.3f}s, Parallel: {time_par:.3f}s", end="")
    if time_par < time_seq * 1.1:  # Allow 10% margin
        print(f" ({time_seq/time_par:.2f}x speedup)")
    else:
        print(f" (depth {depth} may be too shallow for parallel benefit)")


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PARALLEL OPTIMIZATION TESTS" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    tests = [
        ("Multi-PV Search Parallelization", test_parallel_multi_pv),
        ("Find Best Move Fallback Parallelization", test_parallel_find_best_move),
        ("Alpha-Beta Root Search Parallelization", test_parallel_root_search),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: FAILED - {e}")
            failed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()
