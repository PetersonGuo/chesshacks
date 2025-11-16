#!/usr/bin/env python3
"""Test CUDA batch evaluation functionality."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import c_helpers


def test_cuda_detection():
    """Test if CUDA is properly detected."""
    print("=" * 60)
    print("CUDA Detection Test")
    print("=" * 60)

    is_available = c_helpers.is_cuda_available()
    info = c_helpers.get_cuda_info()

    print(f"CUDA Available: {is_available}")
    print(f"CUDA Info: {info}")
    print()

    return is_available


def test_basic_evaluation():
    """Test basic evaluation function."""
    print("=" * 60)
    print("Basic Evaluation Test")
    print("=" * 60)

    # Starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    score = c_helpers.evaluate_with_pst(fen)

    print(f"Starting position: {fen}")
    print(f"Evaluation score: {score}")
    print(f"Expected: ~0 (balanced position)")
    print()

    # White advantage position
    fen2 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    score2 = c_helpers.evaluate_with_pst(fen2)

    print(f"E4 E5 position: {fen2}")
    print(f"Evaluation score: {score2}")
    print()


def test_alpha_beta_cuda():
    """Test alpha-beta search with CUDA."""
    print("=" * 60)
    print("Alpha-Beta CUDA Search Test")
    print("=" * 60)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    depth = 3

    print(f"Position: {fen}")
    print(f"Search depth: {depth}")
    print("Running alpha-beta search with CUDA...")

    # Create search parameters
    tt = c_helpers.TranspositionTable()
    killers = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()

    # Use CPU evaluation function for comparison
    def evaluate(fen_str):
        return c_helpers.evaluate_with_pst(fen_str)

    try:
        score = c_helpers.alpha_beta_cuda(
            fen,
            depth,
            c_helpers.MIN,
            c_helpers.MAX,
            True,
            evaluate,
            tt,
            killers,
            history,
        )
        print(f"Search completed!")
        print(f"Best score: {score}")
        print()

        return score
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all CUDA tests."""
    print("\n" + "=" * 60)
    print("ChessHacks CUDA Test Suite")
    print("=" * 60 + "\n")

    # Test 1: CUDA detection
    cuda_available = test_cuda_detection()

    if not cuda_available:
        print("WARNING: CUDA not available - GPU acceleration disabled")
        print("This is expected if you don't have a CUDA-capable GPU")
        return

    # Test 2: Basic evaluation
    test_basic_evaluation()

    # Test 3: Alpha-beta with CUDA
    test_alpha_beta_cuda()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
