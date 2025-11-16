"""
NNUE (Efficiently Updatable Neural Network) Model Architecture
Bitmap-based variant for chess position evaluation - optimized for fast inference
"""

import torch
import torch.nn as nn
import chess
from typing import Optional


class ClippedReLU(nn.Module):
    """Clipped ReLU activation: min(max(x, 0), 1)"""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class BitboardFeatures:
    """
    Virgo-style bitboard representation for NNUE
    Uses [2 colors][6 piece types] organization for efficient representation

    Follows the Virgo chess engine bitboard structure:
    - pieces[BLACK][PAWN], pieces[BLACK][KNIGHT], ..., pieces[BLACK][KING]
    - pieces[WHITE][PAWN], pieces[WHITE][KNIGHT], ..., pieces[WHITE][KING]
    """
    # Piece type indices (Virgo-style)
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5

    # Player indices (Virgo-style)
    BLACK = 0
    WHITE = 1

    # Map chess.py constants to our indices
    PIECE_TO_INDEX = {
        chess.PAWN: PAWN,
        chess.KNIGHT: KNIGHT,
        chess.BISHOP: BISHOP,
        chess.ROOK: ROOK,
        chess.QUEEN: QUEEN,
        chess.KING: KING,
    }

    COLOR_TO_INDEX = {
        chess.BLACK: BLACK,
        chess.WHITE: WHITE,
    }

    NUM_PIECE_TYPES = 6
    NUM_COLORS = 2
    NUM_SQUARES = 64

    # Precompute mirror indices for 180° rotation (side-to-move perspective)
    MIRROR_INDICES = [chess.square_mirror(square) for square in chess.SQUARES]

    # Total feature size: 2 colors × 6 pieces × 64 squares = 768 features
    FEATURE_SIZE = NUM_COLORS * NUM_PIECE_TYPES * NUM_SQUARES

    @staticmethod
    def board_to_bitmap(board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to Virgo-style bitmap representation

        Args:
            board: python-chess Board object

        Returns:
            Tensor of shape [2, 6, 64] representing bitboards in Virgo format:
            - bitboards[color][piece_type][square] = 1 if piece present, 0 otherwise
            - color: BLACK=0, WHITE=1
            - piece_type: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
        """
        # Initialize empty bitboards: [2 colors, 6 pieces, 64 squares]
        bitboards = torch.zeros(
            BitboardFeatures.NUM_COLORS,
            BitboardFeatures.NUM_PIECE_TYPES,
            BitboardFeatures.NUM_SQUARES,
            dtype=torch.float32
        )

        # Iterate through all squares
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Get color and piece type indices (Virgo-style)
            color_idx = BitboardFeatures.COLOR_TO_INDEX[piece.color]
            piece_idx = BitboardFeatures.PIECE_TO_INDEX[piece.piece_type]

            # Set bit: pieces[color][piece_type][square]
            bitboards[color_idx, piece_idx, square] = 1.0

        return bitboards

    @staticmethod
    def board_to_features(board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to flattened bitmap features

        Args:
            board: python-chess Board object

        Returns:
            Flattened bitmap tensor of shape [768] (2 × 6 × 64)
            Layout: [BLACK pieces (all types), WHITE pieces (all types)]
        """
        bitboards = BitboardFeatures.board_to_bitmap(board)
        return bitboards.flatten()

    @staticmethod
    def get_piece_bitboard(board: chess.Board, color: chess.Color, piece_type: chess.PieceType) -> torch.Tensor:
        """
        Get bitboard for a specific piece type and color (Virgo-style accessor)

        Args:
            board: python-chess Board object
            color: chess.BLACK or chess.WHITE
            piece_type: chess.PAWN, chess.KNIGHT, etc.

        Returns:
            Tensor of shape [64] with 1s where pieces are located
        """
        bitboards = BitboardFeatures.board_to_bitmap(board)
        color_idx = BitboardFeatures.COLOR_TO_INDEX[color]
        piece_idx = BitboardFeatures.PIECE_TO_INDEX[piece_type]
        return bitboards[color_idx, piece_idx]

    @staticmethod
    def board_to_features_for_side(board: chess.Board, perspective: Optional[chess.Color] = None) -> torch.Tensor:
        """
        Convert chess board to flattened bitmap features with colors ordered from the specified perspective.

        Args:
            board: python-chess Board object
            perspective: chess.WHITE or chess.BLACK (defaults to board.turn)

        Returns:
            Flattened bitmap tensor of shape [768] where the first 384 features correspond to
            the perspective player (board rotated so the mover is always at the bottom)
            and the remaining 384 to the opponent.
        """
        if perspective is None:
            perspective = board.turn

        bitboards = BitboardFeatures.board_to_bitmap(board)

        if perspective == chess.WHITE:
            ordered = bitboards[[BitboardFeatures.WHITE, BitboardFeatures.BLACK], :, :]
        else:
            ordered = bitboards[[BitboardFeatures.BLACK, BitboardFeatures.WHITE], :, :]
            ordered = ordered[:, :, BitboardFeatures.MIRROR_INDICES]

        return ordered.flatten()


class ChessNNUEModel(nn.Module):
    """
    NNUE Neural Network for chess position evaluation
    Architecture: Bitmap features (768) → 256 → 32 → 32 → 1 (linear output)

    Uses bitmap/bitboard representation for efficient chess position encoding.
    12 bitboards (6 piece types × 2 colors) × 64 squares = 768 input features.

    Output is linear (no activation) to support full range of normalized evaluations.
    """

    def __init__(self, hidden_size=256, hidden2_size=32, hidden3_size=32):
        super(ChessNNUEModel, self).__init__()

        self.input_size = BitboardFeatures.FEATURE_SIZE
        self.hidden_size = hidden_size

        # Feature transformer: converts bitmap features to dense representation
        self.ft = nn.Linear(self.input_size, hidden_size)

        # Hidden layers (dense)
        self.fc1 = nn.Linear(hidden_size, hidden2_size)
        self.fc2 = nn.Linear(hidden2_size, hidden3_size)
        self.fc3 = nn.Linear(hidden3_size, 1)

        # Activations
        self.crelu = ClippedReLU()

        # Initialize layer weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability and faster convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use gain=1.0 for better initial gradients
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network with bitmap inputs

        Args:
            x: [batch_size, 768] tensor with bitmap features (side-to-move normalized)

        Returns:
            Position evaluation [batch_size, 1] (normalized)
        """
        # Transform features
        x = self.crelu(self.ft(x))

        # Pass through hidden layers
        x = self.crelu(self.fc1(x))
        x = self.crelu(self.fc2(x))
        x = self.fc3(x)

        # Linear output - no activation
        # This allows the model to learn the full range of normalized evaluation scores
        # The loss function (MSE or Huber) will guide learning without artificial bounds
        return x

    def evaluate_board(self, board: chess.Board):
        """
        Evaluate a single chess position

        Args:
            board: python-chess Board object

        Returns:
            Evaluation score (positive = white advantage)
        """
        self.eval()
        with torch.no_grad():
            params = next(self.parameters())
            device = params.device

            # Get bitmap features from side-to-move perspective
            features = BitboardFeatures.board_to_features_for_side(board, perspective=board.turn)
            features = features.unsqueeze(0).to(device)  # Add batch dimension

            # Account for side to move
            flip_sign = -1 if board.turn == chess.BLACK else 1

            # Forward pass
            score = self.forward(features)

            return (score * flip_sign).item()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ChessNNUEModel(
        hidden_size=256,
        hidden2_size=32,
        hidden3_size=32
    )

    print(f"NNUE Model Architecture (Virgo-style Bitboards):")
    print(f"Input size: {model.input_size:,} (2 colors × 6 pieces × 64 squares)")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"\nModel structure:")
    print(model)

    # Test with a starting position
    board = chess.Board()

    # Test Virgo-style bitboard representation
    bitboards = BitboardFeatures.board_to_bitmap(board)
    print(f"\nVirgo-style Bitboard Representation:")
    print(f"Bitboards shape: {bitboards.shape} [colors, pieces, squares]")

    # Show piece counts (Virgo-style access)
    piece_names = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
    for color_name, color in [('Black', chess.BLACK), ('White', chess.WHITE)]:
        print(f"\n{color_name} pieces:")
        for piece_type, piece_name in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                           chess.ROOK, chess.QUEEN, chess.KING], piece_names):
            piece_bb = BitboardFeatures.get_piece_bitboard(board, color, piece_type)
            count = int(piece_bb.sum().item())
            if count > 0:
                print(f"  {piece_name}: {count}")

    # Test flattened features
    features = BitboardFeatures.board_to_features(board)
    print(f"\nFlattened features:")
    print(f"Feature vector shape: {features.shape}")
    print(f"Non-zero features: {(features != 0).sum().item()}")

    # Test evaluation
    eval_score = model.evaluate_board(board)
    print(f"\nBoard evaluation: {eval_score:.2f}")
