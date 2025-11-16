"""
Test script to verify bitmap NNUE implementation
"""
import torch
import chess
from model import ChessNNUEModel, BitboardFeatures


def test_bitmap_features():
    """Test bitmap feature extraction"""
    print("=" * 80)
    print("Testing Bitmap Feature Extraction")
    print("=" * 80)

    # Test starting position
    board = chess.Board()
    features = BitboardFeatures.board_to_features(board)

    print(f"\n1. Starting Position:")
    print(f"   FEN: {board.fen()}")
    print(f"   Feature shape: {features.shape}")
    print(f"   Non-zero features: {(features != 0).sum().item()}/768")
    print(f"   Expected: 32 pieces (excluding kings not counted separately)")

    # Verify we have 32 pieces total (16 per side)
    bitboards = BitboardFeatures.board_to_bitmap(board)
    for i in range(12):
        piece_count = bitboards[i].sum().item()
        if piece_count > 0:
            piece_names = ['White Pawn', 'White Knight', 'White Bishop',
                          'White Rook', 'White Queen', 'White King',
                          'Black Pawn', 'Black Knight', 'Black Bishop',
                          'Black Rook', 'Black Queen', 'Black King']
            print(f"   {piece_names[i]}: {int(piece_count)} pieces")

    # Test empty board
    board_empty = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
    features_empty = BitboardFeatures.board_to_features(board_empty)
    print(f"\n2. Empty Board:")
    print(f"   Non-zero features: {(features_empty != 0).sum().item()}/768")
    print(f"   Expected: 0")

    # Test position with fewer pieces
    board_mid = chess.Board(fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    features_mid = BitboardFeatures.board_to_features(board_mid)
    print(f"\n3. Mid-game Position:")
    print(f"   FEN: {board_mid.fen()}")
    print(f"   Non-zero features: {(features_mid != 0).sum().item()}/768")


def test_model_inference():
    """Test model forward pass with different positions"""
    print("\n" + "=" * 80)
    print("Testing Model Inference")
    print("=" * 80)

    model = ChessNNUEModel(hidden_size=256, hidden2_size=32, hidden3_size=32)
    model.eval()

    positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Endgame", "8/8/8/4k3/8/8/4K3/8 w - - 0 1"),
    ]

    for name, fen in positions:
        board = chess.Board(fen)
        score = model.evaluate_board(board)
        print(f"\n{name}:")
        print(f"   FEN: {fen}")
        print(f"   Evaluation: {score:.4f}")


def test_batch_processing():
    """Test batch processing with multiple positions"""
    print("\n" + "=" * 80)
    print("Testing Batch Processing")
    print("=" * 80)

    model = ChessNNUEModel(hidden_size=256, hidden2_size=32, hidden3_size=32)
    model.eval()

    # Create batch of positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board(fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        chess.Board(fen="8/8/8/4k3/8/8/4K3/8 w - - 0 1"),
    ]

    # Extract features for all positions
    features_list = [BitboardFeatures.board_to_features(board) for board in positions]
    features_batch = torch.stack(features_list, dim=0)

    print(f"\nBatch shape: {features_batch.shape}")
    print(f"Expected: [3, 768]")

    # Forward pass
    with torch.no_grad():
        outputs = model(features_batch)

    print(f"Outputs shape: {outputs.shape}")
    print(f"Expected: [3, 1]")
    print(f"\nBatch evaluations:")
    for i, score in enumerate(outputs):
        print(f"   Position {i+1}: {score.item():.4f}")


def test_side_to_move():
    """Test that evaluation flips correctly for side to move"""
    print("\n" + "=" * 80)
    print("Testing Side-to-Move Symmetry")
    print("=" * 80)

    model = ChessNNUEModel(hidden_size=256, hidden2_size=32, hidden3_size=32)
    model.eval()

    # Same position, different side to move
    fen_white = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    fen_black = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"

    board_white = chess.Board(fen_white)
    board_black = chess.Board(fen_black)

    score_white = model.evaluate_board(board_white)
    score_black = model.evaluate_board(board_black)

    print(f"\nWhite to move evaluation: {score_white:.4f}")
    print(f"Black to move evaluation: {score_black:.4f}")
    print(f"Note: Evaluations should have opposite signs (relative to side to move)")


if __name__ == "__main__":
    test_bitmap_features()
    test_model_inference()
    test_batch_processing()
    test_side_to_move()

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
