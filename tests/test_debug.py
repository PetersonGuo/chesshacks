#!/usr/bin/env python3
"""Debug FEN matching."""

import sys

sys.path.insert(0, "/home/petersonguo/chesshacks/build")
import c_helpers

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print("=" * 80)
print("Creating chess board and generating moves...")
board = c_helpers.ChessBoard(STARTING_FEN)
moves = board.generate_legal_moves()
print(f"Found {len(moves)} legal moves")

print("\nGenerating child FENs:")
for i, move in enumerate(moves[:5]):  # Just first 5
    board.make_move(move)
    child_fen = board.to_fen()
    print(f"{i+1}. {child_fen}")
    board.unmake_move(move)

print("\n" + "=" * 80)
print("Getting best move FEN from find_best_move...")
best_fen = c_helpers.find_best_move(STARTING_FEN, 2, c_helpers.evaluate_with_pst)
print(f"Best move FEN: {best_fen}")
print(f"Best move FEN length: {len(best_fen)}")

# Let's manually check what find_best_move returns character by character
print("\nCharacter breakdown of best_fen:")
for i, c in enumerate(best_fen):
    print(f"[{i}]: '{c}' (ord={ord(c)})")
