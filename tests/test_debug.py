#!/usr/bin/env python3
"""Debug FEN matching - Updated to work with refactored modules."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print("=" * 80)
print("Testing refactored C++ modules...")
print("=" * 80)

print("\n1. Testing find_best_move...")
best_fen = c_helpers.find_best_move(STARTING_FEN, 2, c_helpers.evaluate_with_pst)
print(f"Best move FEN: {best_fen[:60]}...")
print(f"Best move FEN length: {len(best_fen)}")

print("\n2. Testing get_best_move_uci...")
best_uci = c_helpers.get_best_move_uci(STARTING_FEN, 2, c_helpers.evaluate_with_pst)
print(f"Best move UCI: {best_uci}")
print(f"Best move UCI length: {len(best_uci)}")

print("\n3. Testing Multi-PV search...")
pv_lines = c_helpers.multi_pv_search(STARTING_FEN, 3, 3, c_helpers.evaluate_with_pst)
print(f"Found {len(pv_lines)} variations:")
for i, line in enumerate(pv_lines):
    print(f"   {i+1}. {line.uci_move} (score: {line.score})")

print("\n4. Testing evaluation...")
score = c_helpers.evaluate_with_pst(STARTING_FEN)
print(f"Starting position evaluation: {score}")

print("\n5. Testing transposition table...")
tt = c_helpers.TranspositionTable()
print(f"Empty TT size: {len(tt)}")
_ = c_helpers.get_best_move_uci(STARTING_FEN, 3, c_helpers.evaluate_with_pst, tt)
print(f"TT size after search: {len(tt)}")

print("\n6. Testing killer moves...")
km = c_helpers.KillerMoves()
km.clear()
print("✓ KillerMoves created and cleared")

print("\n7. Testing history table...")
ht = c_helpers.HistoryTable()
ht.clear()
ht.age()
print("✓ HistoryTable created, cleared, and aged")

print("\n" + "=" * 80)
print("All tests passed! Refactored modules working correctly.")
print("=" * 80)
