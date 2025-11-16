#!/usr/bin/env python3
"""
Test PGN to FEN conversion functionality
Tests various PGN formats and move sequences
"""

import sys
import os

# Add build directory to path to import c_helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c_helpers


def test_pgn_to_fen():
    """Test PGN to FEN conversion with various game sequences"""
    print("=" * 70)
    print("PGN TO FEN CONVERSION TESTS")
    print("=" * 70)
    print()

    test_cases = [
        {
            "name": "Starting Position (Empty PGN)",
            "pgn": "",
            "expected_piece_placement": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        },
        {
            "name": "Single Move (1. e4)",
            "pgn": "1. e4",
            "expected_piece_placement": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        },
        {
            "name": "Two Moves (1. e4 e5)",
            "pgn": "1. e4 e5",
            "expected_piece_placement": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
        },
        {
            "name": "Italian Game Opening",
            "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4",
            "expected_piece_placement": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
        },
        {
            "name": "Scholar's Mate Setup",
            "pgn": "1. e4 e5 2. Bc4 Nc6 3. Qh5",
            "expected_piece_placement": "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR",
        },
        {
            "name": "Sicilian Defense",
            "pgn": "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4",
            "expected_piece_placement": "rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R",
        },
        {
            "name": "French Defense",
            "pgn": "1. e4 e6 2. d4 d5 3. Nc3",
            "expected_piece_placement": "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR",
        },
        {
            "name": "King's Gambit",
            "pgn": "1. e4 e5 2. f4 exf4",
            "expected_piece_placement": "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR",
        },
        {
            "name": "Castling Kingside",
            "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O",
            "expected_piece_placement": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1",
        },
        {
            "name": "Castling Queenside",
            "pgn": "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. a3 Bxc3+ 6. Qxc3 d6 7. Nf3 Nbd7 8. e3 b6 9. Be2 Bb7 10. O-O",
            "expected_piece_placement": "r2q1rk1/pbpn1ppp/1p1ppn2/8/2PP4/P1Q1PN2/1P2BPPP/R1B2RK1",
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"PGN: {test['pgn'] if test['pgn'] else '(empty - starting position)'}")
        
        try:
            result_fen = c_helpers.pgn_to_fen(test['pgn'])
            
            # Extract piece placement (first field of FEN)
            result_pieces = result_fen.split()[0]
            expected_pieces = test['expected_piece_placement']
            
            if result_pieces == expected_pieces:
                print(f"✓ PASSED")
                print(f"  FEN: {result_fen}")
                passed += 1
            else:
                print(f"✗ FAILED")
                print(f"  Expected: {expected_pieces}")
                print(f"  Got:      {result_pieces}")
                print(f"  Full FEN: {result_fen}")
                failed += 1
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
            
        print()

    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)
    
    return failed == 0


def test_invalid_pgn():
    """Test error handling for invalid PGN"""
    print()
    print("=" * 70)
    print("INVALID PGN ERROR HANDLING")
    print("=" * 70)
    print()
    
    invalid_cases = [
        ("Invalid move notation", "1. xyz"),
        ("Illegal move", "1. e5"),  # Pawn can't move 3 squares
        ("Move to occupied square", "1. Nf3 Nf6 2. Nf3"),  # Knight already on f3
    ]
    
    for name, pgn in invalid_cases:
        print(f"Test: {name}")
        print(f"PGN: {pgn}")
        try:
            result = c_helpers.pgn_to_fen(pgn)
            print(f"  Result: {result}")
            print(f"  Note: Returned result without error")
        except Exception as e:
            print(f"  Exception: {e}")
        print()


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "PGN TO FEN TEST" + " " * 33 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    success = test_pgn_to_fen()
    test_invalid_pgn()
    
    if success:
        print("\n✅ All PGN to FEN conversion tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
