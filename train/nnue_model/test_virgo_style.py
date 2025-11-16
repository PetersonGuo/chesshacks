"""
Test Virgo-style bitboard implementation
Verifies compatibility with the Virgo chess engine bitboard format
"""
import os
import sys

import torch
import chess

try:
    from .model import ChessNNUEModel, BitboardFeatures
except ImportError:
    # Allow running as a standalone script (python train/nnue_model/test_virgo_style.py)
    sys.path.insert(0, os.path.dirname(__file__))
    from model import ChessNNUEModel, BitboardFeatures


def test_virgo_structure():
    """Test that bitboards follow Virgo [2][6][64] structure"""
    print("=" * 80)
    print("Testing Virgo-style Bitboard Structure")
    print("=" * 80)

    board = chess.Board()
    bitboards = BitboardFeatures.board_to_bitmap(board)

    # Verify shape
    assert bitboards.shape == (2, 6, 64), f"Expected (2, 6, 64), got {bitboards.shape}"
    print(f"✓ Bitboard shape: {bitboards.shape} [colors, pieces, squares]")

    # Verify indexing matches Virgo convention
    assert BitboardFeatures.BLACK == 0, "BLACK should be 0"
    assert BitboardFeatures.WHITE == 1, "WHITE should be 1"
    print(f"✓ Color indices: BLACK={BitboardFeatures.BLACK}, WHITE={BitboardFeatures.WHITE}")

    assert BitboardFeatures.PAWN == 0, "PAWN should be 0"
    assert BitboardFeatures.KNIGHT == 1, "KNIGHT should be 1"
    assert BitboardFeatures.BISHOP == 2, "BISHOP should be 2"
    assert BitboardFeatures.ROOK == 3, "ROOK should be 3"
    assert BitboardFeatures.QUEEN == 4, "QUEEN should be 4"
    assert BitboardFeatures.KING == 5, "KING should be 5"
    print(f"✓ Piece indices: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5")

    print(f"\n✓ Virgo-style structure verified!")


def test_piece_access():
    """Test Virgo-style piece accessors"""
    print("\n" + "=" * 80)
    print("Testing Virgo-style Piece Access")
    print("=" * 80)

    board = chess.Board()

    # Test accessing specific piece bitboards
    white_pawns = BitboardFeatures.get_piece_bitboard(board, chess.WHITE, chess.PAWN)
    assert white_pawns.sum() == 8, f"Expected 8 white pawns, got {int(white_pawns.sum())}"
    print(f"✓ White pawns: {int(white_pawns.sum())}")

    black_knights = BitboardFeatures.get_piece_bitboard(board, chess.BLACK, chess.KNIGHT)
    assert black_knights.sum() == 2, f"Expected 2 black knights, got {int(black_knights.sum())}"
    print(f"✓ Black knights: {int(black_knights.sum())}")

    white_king = BitboardFeatures.get_piece_bitboard(board, chess.WHITE, chess.KING)
    assert white_king.sum() == 1, f"Expected 1 white king, got {int(white_king.sum())}"
    print(f"✓ White king: {int(white_king.sum())}")

    print(f"\n✓ Piece accessors working correctly!")


def test_bitboard_positions():
    """Test that bitboard squares match actual piece positions"""
    print("\n" + "=" * 80)
    print("Testing Bitboard Square Positions")
    print("=" * 80)

    board = chess.Board()
    bitboards = BitboardFeatures.board_to_bitmap(board)

    # Verify white pawns are on rank 2 (squares 8-15)
    white_pawns = bitboards[BitboardFeatures.WHITE, BitboardFeatures.PAWN]
    expected_squares = list(range(8, 16))  # a2-h2
    actual_squares = [i for i in range(64) if white_pawns[i] == 1.0]
    assert actual_squares == expected_squares, f"White pawns mismatch: {actual_squares} vs {expected_squares}"
    print(f"✓ White pawns on correct squares: {actual_squares}")

    # Verify black rooks on a8 (56) and h8 (63)
    black_rooks = bitboards[BitboardFeatures.BLACK, BitboardFeatures.ROOK]
    expected_squares = [56, 63]  # a8, h8
    actual_squares = [i for i in range(64) if black_rooks[i] == 1.0]
    assert actual_squares == expected_squares, f"Black rooks mismatch: {actual_squares} vs {expected_squares}"
    print(f"✓ Black rooks on correct squares: {actual_squares}")

    print(f"\n✓ Square positions verified!")


def test_virgo_vs_flat():
    """Compare Virgo-style [2][6][64] with flattened [768]"""
    print("\n" + "=" * 80)
    print("Testing Virgo-style vs Flattened Representation")
    print("=" * 80)

    board = chess.Board()

    # Get both representations
    virgo_style = BitboardFeatures.board_to_bitmap(board)  # [2, 6, 64]
    flattened = BitboardFeatures.board_to_features(board)  # [768]

    # Verify flattening is correct
    assert virgo_style.flatten().shape == flattened.shape, "Shape mismatch"
    assert torch.allclose(virgo_style.flatten(), flattened), "Values don't match"

    print(f"Virgo-style shape: {virgo_style.shape}")
    print(f"Flattened shape: {flattened.shape}")
    print(f"✓ Flattening preserves data correctly")

    # Verify reconstruction
    reconstructed = flattened.reshape(2, 6, 64)
    assert torch.allclose(reconstructed, virgo_style), "Reconstruction failed"
    print(f"✓ Can reconstruct Virgo-style from flattened")

    print(f"\n✓ Format compatibility verified!")


def test_empty_and_full_boards():
    """Test edge cases: empty board and full board"""
    print("\n" + "=" * 80)
    print("Testing Edge Cases")
    print("=" * 80)

    # Empty board
    empty_board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
    empty_bb = BitboardFeatures.board_to_bitmap(empty_board)
    assert empty_bb.sum() == 0, "Empty board should have no pieces"
    print(f"✓ Empty board: {int(empty_bb.sum())} pieces")

    # Starting position
    start_board = chess.Board()
    start_bb = BitboardFeatures.board_to_bitmap(start_board)
    assert start_bb.sum() == 32, "Starting position should have 32 pieces"
    print(f"✓ Starting position: {int(start_bb.sum())} pieces")

    # Mid-game position
    mid_board = chess.Board(fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mid_bb = BitboardFeatures.board_to_bitmap(mid_board)
    print(f"✓ Mid-game position: {int(mid_bb.sum())} pieces")

    print(f"\n✓ Edge cases handled correctly!")


def test_model_compatibility():
    """Test that model works with Virgo-style features"""
    print("\n" + "=" * 80)
    print("Testing Model Compatibility")
    print("=" * 80)

    model = ChessNNUEModel(hidden_size=256, hidden2_size=32, hidden3_size=32)
    model.eval()

    # Test evaluation
    board = chess.Board()
    score = model.evaluate_board(board)
    print(f"✓ Model evaluation: {score:.4f}")

    # Test batch processing
    positions = [
        chess.Board(),
        chess.Board(fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        chess.Board(fen="8/8/8/4k3/8/8/4K3/8 w - - 0 1"),
    ]

    features = torch.stack([BitboardFeatures.board_to_features(b) for b in positions])
    with torch.no_grad():
        outputs = model(features)

    assert outputs.shape == (3, 1), f"Expected (3, 1), got {outputs.shape}"
    print(f"✓ Batch evaluation: {outputs.shape}")

    print(f"\n✓ Model compatibility verified!")


def test_side_to_move_features():
    """Ensure side-to-move perspective reordering works"""
    board = chess.Board()
    base = BitboardFeatures.board_to_features(board).reshape(2, 6, 64)
    mirror_idx = BitboardFeatures.MIRROR_INDICES

    white_first = BitboardFeatures.board_to_features_for_side(board, perspective=chess.WHITE)
    black_first = BitboardFeatures.board_to_features_for_side(board, perspective=chess.BLACK)

    expected_white = torch.stack(
        (base[BitboardFeatures.WHITE], base[BitboardFeatures.BLACK])
    ).flatten()
    expected_black = torch.stack(
        (base[BitboardFeatures.BLACK][:, mirror_idx], base[BitboardFeatures.WHITE][:, mirror_idx])
    ).flatten()

    assert torch.allclose(white_first, expected_white)
    assert torch.allclose(black_first, expected_black)
    print("\n✓ Side-to-move reordering (with mirroring) verified!")


def test_virgo_layout_documentation():
    """Document the exact Virgo-style layout"""
    print("\n" + "=" * 80)
    print("Virgo-style Bitboard Layout Documentation")
    print("=" * 80)

    board = chess.Board()
    bitboards = BitboardFeatures.board_to_bitmap(board)

    print("\nBitboard organization: bitboards[color][piece_type][square]")
    print(f"Shape: {bitboards.shape}")
    print()

    # Show layout
    print("Index mapping:")
    print("  Color dimension [0-1]:")
    print(f"    0 = BLACK")
    print(f"    1 = WHITE")
    print()
    print("  Piece dimension [0-5]:")
    print(f"    0 = PAWN")
    print(f"    1 = KNIGHT")
    print(f"    2 = BISHOP")
    print(f"    3 = ROOK")
    print(f"    4 = QUEEN")
    print(f"    5 = KING")
    print()
    print("  Square dimension [0-63]:")
    print(f"    0-7   = rank 1 (a1-h1)")
    print(f"    8-15  = rank 2 (a2-h2)")
    print(f"    ...")
    print(f"    56-63 = rank 8 (a8-h8)")
    print()

    # Example access patterns
    print("Example access patterns:")
    print(f"  Black pawns:   bitboards[{BitboardFeatures.BLACK}][{BitboardFeatures.PAWN}][:]")
    print(f"  White knights: bitboards[{BitboardFeatures.WHITE}][{BitboardFeatures.KNIGHT}][:]")
    print(f"  Black king:    bitboards[{BitboardFeatures.BLACK}][{BitboardFeatures.KING}][:]")
    print()

    # Flattening order
    print("Flattening to [768]:")
    print("  Order: BLACK pieces (all 6 types × 64 squares), then WHITE pieces (all 6 types × 64 squares)")
    print(f"  Indices 0-383:   BLACK pieces")
    print(f"  Indices 384-767: WHITE pieces")
    print()

    print("✓ Layout documented!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VIRGO-STYLE BITBOARD IMPLEMENTATION TESTS")
    print("=" * 80)

    try:
        test_virgo_structure()
        test_piece_access()
        test_bitboard_positions()
        test_virgo_vs_flat()
        test_empty_and_full_boards()
        test_model_compatibility()
        test_virgo_layout_documentation()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nVirgo-style bitboard implementation is working correctly!")
        print("The [2][6][64] structure matches the Virgo chess engine format.")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
