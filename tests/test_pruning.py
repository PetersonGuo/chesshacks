#!/usr/bin/env python3
"""Test the newly implemented pruning features."""

import sys
import time

sys.path.insert(0, "build")
import c_helpers

# Starting position FEN
STARTING_POSITION_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

print("Testing Advanced Pruning Features")
print("=" * 70)

# Test 1: Razoring, Futility Pruning, and SEE
print("\n1. Testing move selection with all features...")
transposition_table = c_helpers.TranspositionTable()
killer_moves = c_helpers.KillerMoves()
history_table = c_helpers.HistoryTable()

test_depths = [3, 4, 5]
for search_depth in test_depths:
    start_time = time.time()
    best_move_uci = c_helpers.get_best_move_uci(
        STARTING_POSITION_FEN, search_depth, c_helpers.evaluate_with_pst, 
        transposition_table, 0, killer_moves, history_table
    )
    elapsed_time = time.time() - start_time
    tt_entries_count = len(transposition_table)
    print(f"   Depth {search_depth}: {best_move_uci} ({elapsed_time:.3f}s, {tt_entries_count} TT entries)")

# Test 2: Tactical position with captures (SEE should help)
print("\n2. Testing tactical position with SEE...")
# Position with multiple captures
tactical_position_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
transposition_table.clear()

start_time = time.time()
best_tactical_move = c_helpers.get_best_move_uci(
    tactical_position_fen, 5, c_helpers.evaluate_with_pst, transposition_table
)
elapsed_time = time.time() - start_time
tt_entries_count = len(transposition_table)
print(f"   Best move: {best_tactical_move} ({elapsed_time:.3f}s)")
print(f"   TT entries: {tt_entries_count}")

# Test 3: Position where razoring should help (clearly losing)
print("\n3. Testing razoring on losing position...")
losing_position_fen = "rnbqkb1r/pppppppp/8/8/8/8/1PPPPPPP/RNBQKB1R w KQkq - 0 1"  # Down a knight
transposition_table.clear()

start_time = time.time()
position_score = c_helpers.alpha_beta_optimized(
    losing_position_fen, 4, c_helpers.MIN, c_helpers.MAX, True, 
    c_helpers.evaluate_with_pst, transposition_table
)
elapsed_time = time.time() - start_time
tt_entries_count = len(transposition_table)
print(f"   Score: {position_score} ({elapsed_time:.3f}s, {tt_entries_count} TT entries)")
print(f"   (Razoring should speed up clearly bad positions)")

print("\n" + "=" * 70)
print("Pruning tests completed!")
print("\nNew Features Tested:")
print("  • Razoring - prunes at low depths when eval << alpha")
print("  • Futility Pruning - skips quiet moves unlikely to help")
print("  • Static Exchange Evaluation (SEE) - better capture ordering")
