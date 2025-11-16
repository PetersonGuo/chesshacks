#!/usr/bin/env python3
"""Test the new C++ best move retrieval functions."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print("Testing C++ Best Move Retrieval")
print("=" * 60)

# Test 1: Get best move from starting position
print("\n1. Starting Position")
print(f"   FEN: {STARTING_FEN}")
print("   Searching at depth 4...")

best_move_fen = c_helpers.find_best_move(STARTING_FEN, 4, c_helpers.evaluate_with_pst)
print(f"   Best move FEN: {best_move_fen[:50]}...")

best_move_uci = c_helpers.get_best_move_uci(
    STARTING_FEN, 4, c_helpers.evaluate_with_pst
)
print(f"   Best move UCI: {best_move_uci}")

# Test 2: A tactical position (Scholar's Mate threat)
print("\n2. Tactical Position (Scholar's Mate)")
tactical_fen = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
print(f"   FEN: {tactical_fen}")
print("   Searching at depth 3...")

best_move_uci = c_helpers.get_best_move_uci(
    tactical_fen, 3, c_helpers.evaluate_with_pst
)
print(f"   Best move UCI: {best_move_uci}")

# Test 3: Endgame position
print("\n3. Simple Endgame")
endgame_fen = "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"
print(f"   FEN: {endgame_fen}")
print("   Searching at depth 4...")

best_move_uci = c_helpers.get_best_move_uci(endgame_fen, 4, c_helpers.evaluate_with_pst)
print(f"   Best move UCI: {best_move_uci}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
