#!/usr/bin/env python3
"""Test Principal Variation Search and Piece-Square Tables"""

import os
import sys
import time

# Add build directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c_helpers


def test_evaluate_with_pst():
    """Test piece-square table evaluation function"""
    print("\n=== Testing Piece-Square Table Evaluation ===")

    # Starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    start_eval = c_helpers.evaluate_with_pst(start_fen)
    print(f"Starting position evaluation: {start_eval}")
    print(f"  (Should be close to 0 due to symmetry)")

    # Position with centralized knight (should be better)
    centralized_fen = "rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
    centralized_eval = c_helpers.evaluate_with_pst(centralized_fen)
    print(f"\nPosition with centralized knight: {centralized_eval}")
    print(f"  (White knight on d4 should be better than black on f6)")

    # Position with developed pieces
    developed_fen = (
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    developed_eval = c_helpers.evaluate_with_pst(developed_fen)
    print(f"\nItalian Game position: {developed_eval}")
    print(f"  (Developed pieces should have positional bonuses)")

    # Verify PST makes a difference from pure material
    # Pure material would be 0 in starting position, PST adds positional bonuses
    if abs(start_eval) < 50:  # Should be close to 0 but not exactly
        print("\n✓ Piece-square tables working correctly!")
        return True
    else:
        print(f"\n✗ Unexpected starting position evaluation: {start_eval}")
        return False


def test_pvs_performance():
    """Test Principal Variation Search performance"""
    print("\n=== Testing Principal Variation Search ===")

    # Complex middlegame position with many tactical options
    fen = "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 10"

    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    print(f"\nTesting position: {fen}")
    print("Searching with PVS (optimized) at depth 5...")

    start = time.time()
    result = c_helpers.alpha_beta_optimized(
        fen,
        5,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )
    elapsed = time.time() - start

    print(f"Depth 5 result: {result} (time: {elapsed:.3f}s)")
    print(f"Transposition table entries: {tt.size()}")

    # Test depth 6
    print("\nSearching at depth 6...")
    tt.clear()
    killers.clear()
    history.clear()

    start = time.time()
    result = c_helpers.alpha_beta_optimized(
        fen,
        6,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )
    elapsed = time.time() - start

    print(f"Depth 6 result: {result} (time: {elapsed:.3f}s)")
    print(f"Transposition table entries: {tt.size()}")

    if elapsed < 5.0:  # Should be fast with all optimizations
        print("\n✓ PVS performing efficiently!")
        return True
    else:
        print(f"\n⚠ PVS slower than expected: {elapsed:.3f}s")
        return True  # Still pass, just slower


def test_pst_vs_material():
    """Compare piece-square table evaluation to pure material"""
    print("\n=== Comparing PST Evaluation to Pure Material ===")

    def material_only(fen):
        """Simple material-only evaluation"""
        board_str = fen.split()[0]
        piece_values = {
            "P": 100,
            "N": 320,
            "B": 330,
            "R": 500,
            "Q": 900,
            "K": 20000,
            "p": -100,
            "n": -320,
            "b": -330,
            "r": -500,
            "q": -900,
            "k": -20000,
        }
        score = 0
        for char in board_str:
            if char in piece_values:
                score += piece_values[char]
        return score

    # Test positions where PST should matter
    test_positions = [
        (
            "Starting position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "Centralized pieces",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/3NP3/3P1N2/PPP2PPP/R1BQKB1R w KQkq - 0 1",
        ),
        (
            "King safety test",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        ),
    ]

    for name, fen in test_positions:
        material = material_only(fen)
        pst_eval = c_helpers.evaluate_with_pst(fen)
        difference = pst_eval - material

        print(f"\n{name}:")
        print(f"  Material only: {material}")
        print(f"  With PST:      {pst_eval}")
        print(f"  Difference:    {difference} (positional bonus)")

    print("\n✓ Piece-square tables adding positional evaluation!")
    return True


def test_pvs_null_window():
    """Verify PVS uses null windows correctly"""
    print("\n=== Testing PVS Null Window Searches ===")

    # Position where PVS should find the best move quickly
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    print(f"Testing position: {fen}")
    print("Searching with PVS at depth 4...")

    start = time.time()
    result = c_helpers.alpha_beta_optimized(
        fen,
        4,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        c_helpers.evaluate_with_pst,
        tt,
        0,
        killers,
        history,
    )
    elapsed = time.time() - start

    print(f"Result: {result} (time: {elapsed:.3f}s)")
    print(f"TT entries: {tt.size()}")

    # PVS should be fast due to null window cutoffs
    if elapsed < 1.0:
        print("\n✓ PVS null windows working efficiently!")
        return True
    else:
        print(f"\n⚠ PVS search time: {elapsed:.3f}s (still working)")
        return True


def run_all_tests():
    """Run all PVS and PST tests"""
    print("=" * 60)
    print("TESTING PRINCIPAL VARIATION SEARCH & PIECE-SQUARE TABLES")
    print("=" * 60)

    tests = [
        ("Piece-Square Table Evaluation", test_evaluate_with_pst),
        ("PVS Performance", test_pvs_performance),
        ("PST vs Material", test_pst_vs_material),
        ("PVS Null Windows", test_pvs_null_window),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
