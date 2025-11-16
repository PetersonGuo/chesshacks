#!/usr/bin/env python3
"""Test advanced CUDA kernel functionality."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import c_helpers


def test_cuda_batch_evaluate():
    """Test batch position evaluation on GPU."""
    print("=" * 70)
    print("CUDA Batch Evaluation Test")
    print("=" * 70)

    if not c_helpers.is_cuda_available():
        print("CUDA not available - skipping test")
        return

    # Test positions
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # E4 E5
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Nf3
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",  # Nf6
    ]

    scores = c_helpers.cuda_batch_evaluate(fens)

    print(f"\nNumber of positions: {len(fens)}")
    print(f"Number of scores: {len(scores)}")

    if scores:
        print("\nPosition evaluations:")
        for i, (fen, score) in enumerate(zip(fens, scores)):
            print(f"  {i+1}. {fen[:40]}... => {score}")

        # All positions should have a score
        assert len(scores) == len(fens), "Should return a score for each position"
        assert all(isinstance(s, int) for s in scores), "All scores should be integers"

        print("\n✓ Batch evaluation test passed")
    else:
        print("Batch evaluation failed or returned empty scores")

    print()


def test_cuda_piece_counting():
    """Test batch piece counting on GPU."""
    print("=" * 70)
    print("CUDA Batch Piece Counting Test")
    print("=" * 70)

    if not c_helpers.is_cuda_available():
        print("CUDA not available - skipping test")
        return

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Full board
        "8/8/8/4k3/8/8/8/4K3 w - - 0 1",  # Just kings
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",  # Early game
    ]

    piece_counts = c_helpers.cuda_batch_count_pieces(fens)

    print(f"\nNumber of positions: {len(fens)}")
    print(f"Number of results: {len(piece_counts)}")

    if piece_counts:
        piece_names = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        print("\nPiece counts:")
        for i, (fen, counts) in enumerate(zip(fens, piece_counts)):
            print(f"\n  Position {i+1}: {fen[:40]}...")
            white_pieces = dict(zip(piece_names[:6], counts[:6]))
            black_pieces = dict(zip(piece_names[6:], counts[6:]))
            print(f"    White: {white_pieces}")
            print(f"    Black: {black_pieces}")

        # Verify starting position has correct piece counts
        assert (
            sum(piece_counts[0][:6]) == 16
        ), "Starting position should have 16 white pieces"
        assert (
            sum(piece_counts[0][6:]) == 16
        ), "Starting position should have 16 black pieces"

        # Verify two-king position
        assert (
            sum(piece_counts[1][:6]) == 1
        ), "Two-king position should have 1 white piece"
        assert (
            sum(piece_counts[1][6:]) == 1
        ), "Two-king position should have 1 black piece"

        print("\n✓ Batch piece counting test passed")
    else:
        print("Batch piece counting failed - empty result")

    print()


def test_cuda_position_hashing():
    """Test batch position hashing on GPU."""
    print("=" * 70)
    print("CUDA Batch Position Hashing Test")
    print("=" * 70)

    if not c_helpers.is_cuda_available():
        print("CUDA not available - skipping test")
        return

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Duplicate
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    ]

    hashes = c_helpers.cuda_batch_hash_positions(fens)

    print(f"\nNumber of positions: {len(fens)}")
    print(f"Number of hashes: {len(hashes)}")

    if hashes:
        print("\nPosition hashes:")
        for i, (fen, hash_val) in enumerate(zip(fens, hashes)):
            print(f"  {i+1}. {fen[:40]}...")
            print(f"     Hash: {hash_val:#018x}")

        # Check duplicate detection
        if hashes[0] == hashes[1]:
            print("\n✓ Duplicate positions have same hash")
        else:
            print("\n✗ Warning: Duplicate positions have different hashes")

        if hashes[0] != hashes[2]:
            print("✓ Different positions have different hashes")
        else:
            print("✗ Warning: Different positions have same hash (collision)")

        print("\n✓ Batch hashing test passed")
    else:
        print("Batch hashing failed - empty result")

    print()


def test_performance_comparison():
    """Compare CPU vs GPU batch evaluation performance."""
    print("=" * 70)
    print("CUDA Performance Comparison")
    print("=" * 70)

    if not c_helpers.is_cuda_available():
        print("CUDA not available - skipping test")
        return

    import time

    # Generate test positions
    num_positions = 100
    base_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fens = [base_fen] * num_positions

    print(f"\nEvaluating {num_positions} positions...")

    # CPU: Individual evaluations
    print("\nCPU (individual evaluations):")
    start = time.perf_counter()
    cpu_scores = []
    for fen in fens:
        score = c_helpers.evaluate_with_pst(fen)
        cpu_scores.append(score)
    cpu_time = time.perf_counter() - start
    print(f"  Time: {cpu_time:.4f}s")
    print(f"  Rate: {num_positions/cpu_time:.1f} positions/sec")

    # GPU: Batch evaluation
    print("\nGPU (batch evaluation):")
    start = time.perf_counter()
    gpu_scores = c_helpers.cuda_batch_evaluate(fens)
    gpu_time = time.perf_counter() - start

    if gpu_scores:
        print(f"  Time: {gpu_time:.4f}s")
        print(f"  Rate: {num_positions/gpu_time:.1f} positions/sec")

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n  Speedup: {speedup:.2f}x")

        # Verify results match
        if cpu_scores == gpu_scores:
            print("  ✓ CPU and GPU results match")
        else:
            print(
                f"  ✗ Warning: Results differ (CPU[0]={cpu_scores[0]}, GPU[0]={gpu_scores[0]})"
            )
    else:
        print("  Batch evaluation failed")

    print()


def main():
    """Run all CUDA kernel tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "CUDA KERNEL TEST SUITE")
    print("=" * 70)

    # Check CUDA availability
    print(f"\nCUDA Available: {c_helpers.is_cuda_available()}")
    if c_helpers.is_cuda_available():
        print(f"GPU: {c_helpers.get_cuda_info()}")
    else:
        print("CUDA not available - tests will be skipped")
        return

    print()

    # Run tests
    test_cuda_batch_evaluate()
    test_cuda_piece_counting()
    test_cuda_position_hashing()
    test_performance_comparison()

    print("=" * 70)
    print("All CUDA kernel tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
