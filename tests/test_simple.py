#!/usr/bin/env python3
"""Simple test for debugging."""

import sys

sys.path.insert(0, "/home/petersonguo/chesshacks/build")
import c_helpers

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print("Testing find_best_move...")
best_fen = c_helpers.find_best_move(STARTING_FEN, 3, c_helpers.evaluate_with_pst)
print(f"Result: '{best_fen}'")
print(f"Length: {len(best_fen)}")
print()
Í
print("Testing get_best_move_uci...")
best_uci = c_helpers.get_best_move_uci(STARTING_FEN, 3, c_helpers.evaluate_with_pst)
print(f"Result: '{best_uci}'")
print(f"Length: {len(best_uci)}")
