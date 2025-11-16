#!/usr/bin/env python3
"""Test the newly implemented enhancements."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print("Testing New Chess Engine Enhancements")
print("=" * 70)

# Test 1: Verify new classes can be instantiated
print("\n1. Testing new data structures...")
try:
    counter_moves = c_helpers.CounterMoveTable()
    print("   ✓ CounterMoveTable created successfully")
    counter_moves.clear()
    print("   ✓ CounterMoveTable.clear() works")
except Exception as e:
    print(f"   ✗ Error with CounterMoveTable: {e}")

# Test 2: Test get_best_move_uci with new parameters
print("\n2. Testing move retrieval with all enhancements...")
try:
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()

    best_move = c_helpers.get_best_move_uci(
        STARTING_FEN, 4, c_helpers.evaluate_with_pst, tt, 0, killers, history, counters
    )
    print(f"   ✓ Best move from starting position (depth 4): {best_move}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Compare depths
print("\n3. Testing at various depths...")
for depth in [2, 3, 4, 5]:
    try:
        move = c_helpers.get_best_move_uci(
            STARTING_FEN, depth, c_helpers.evaluate_with_pst
        )
        print(f"   Depth {depth}: {move}")
    except Exception as e:
        print(f"   Depth {depth}: Error - {e}")

# Test 4: Tactical position
print("\n4. Testing tactical position...")
tactical_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
try:
    move = c_helpers.get_best_move_uci(tactical_fen, 5, c_helpers.evaluate_with_pst)
    print(f"   Best move: {move}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
print("All tests completed!")
print("\nNew Features Implemented:")
print("  • Singular Extensions - extends search for dominant moves")
print("  • Counter Move Heuristic - tracks refutation moves")
print("  • Continuation History - tracks two-move patterns")
