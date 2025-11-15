"""
NNUE (Efficiently Updatable Neural Network) Model Architecture
HalfKP variant for chess position evaluation
"""

import torch
import torch.nn as nn
import chess


class ClippedReLU(nn.Module):
    """Clipped ReLU activation: min(max(x, 0), 1)"""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class HalfKP:
    """
    HalfKP feature representation for NNUE
    Features: King position + Piece positions (Half of the board from king's perspective)
    """
    # Piece type indices (excluding king)
    PIECE_TO_INDEX = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }

    NUM_PIECE_TYPES = 10  # 5 piece types × 2 colors
    NUM_SQUARES = 64
    KING_SQUARES = 64

    # Total feature size: 64 (king squares) × 64 (piece squares) × 10 (piece types)
    FEATURE_SIZE = KING_SQUARES * NUM_SQUARES * NUM_PIECE_TYPES

    @staticmethod
    def get_feature_index(king_square, piece_square, piece_type, piece_color):
        """
        Calculate the feature index for a piece relative to king position

        Args:
            king_square: Square index of the king (0-63)
            piece_square: Square index of the piece (0-63)
            piece_type: chess.PAWN, KNIGHT, etc.
            piece_color: chess.WHITE or chess.BLACK

        Returns:
            Feature index in the input vector
        """
        piece_idx = HalfKP.PIECE_TO_INDEX[piece_type]
        if piece_color == chess.BLACK:
            piece_idx += 5

        return king_square * (HalfKP.NUM_SQUARES * HalfKP.NUM_PIECE_TYPES) + \
               piece_square * HalfKP.NUM_PIECE_TYPES + \
               piece_idx

    @staticmethod
    def board_to_features(board: chess.Board):
        """
        Convert chess board to HalfKP feature vector

        Args:
            board: python-chess Board object

        Returns:
            Tuple of (white_features, black_features) as sparse indices
        """
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        white_features = []
        black_features = []

        # Iterate through all pieces on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue

            # Features from white's perspective
            white_idx = HalfKP.get_feature_index(
                white_king_sq, square, piece.piece_type, piece.color
            )
            white_features.append(white_idx)

            # Features from black's perspective (flip the board)
            black_square = chess.square_mirror(square)
            black_king_sq_flipped = chess.square_mirror(black_king_sq)
            black_idx = HalfKP.get_feature_index(
                black_king_sq_flipped, black_square, piece.piece_type,
                not piece.color  # Flip color from black's perspective
            )
            black_features.append(black_idx)

        return white_features, black_features


class NNUEModel(nn.Module):
    """
    NNUE Neural Network for chess position evaluation
    Architecture: HalfKP features → 256 → 32 → 32 → 1
    """

    def __init__(self, hidden_size=256, hidden2_size=32, hidden3_size=32):
        super(NNUEModel, self).__init__()

        self.input_size = HalfKP.FEATURE_SIZE
        self.hidden_size = hidden_size

        # Feature transformer: converts sparse HalfKP features to dense representation
        # We have two perspectives (white and black), each gets transformed separately
        self.ft = nn.Linear(self.input_size, hidden_size)

        # Hidden layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden2_size)  # *2 because we concat white+black
        self.fc2 = nn.Linear(hidden2_size, hidden3_size)
        self.fc3 = nn.Linear(hidden3_size, 1)

        # Activations
        self.crelu = ClippedReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, white_features, black_features):
        """
        Forward pass through the network

        Args:
            white_features: Sparse feature tensor from white's perspective [batch, feature_size]
            black_features: Sparse feature tensor from black's perspective [batch, feature_size]

        Returns:
            Position evaluation in centipawns [batch, 1]
        """
        # Transform features from both perspectives
        w = self.crelu(self.ft(white_features))
        b = self.crelu(self.ft(black_features))

        # Concatenate both perspectives
        x = torch.cat([w, b], dim=1)

        # Pass through hidden layers
        x = self.crelu(self.fc1(x))
        x = self.crelu(self.fc2(x))
        x = self.fc3(x)

        return x

    def evaluate_board(self, board: chess.Board):
        """
        Evaluate a single chess position

        Args:
            board: python-chess Board object

        Returns:
            Evaluation score in centipawns (positive = white advantage)
        """
        self.eval()
        with torch.no_grad():
            params = next(self.parameters())
            device = params.device
            dtype = params.dtype

            # Get sparse features
            white_idx, black_idx = HalfKP.board_to_features(board)

            # Convert to dense feature vectors
            white_features = torch.zeros(1, self.input_size, device=device, dtype=dtype)
            black_features = torch.zeros(1, self.input_size, device=device, dtype=dtype)

            white_features[0, white_idx] = 1.0
            black_features[0, black_idx] = 1.0

            if board.turn == chess.BLACK:
                white_features, black_features = black_features, white_features
                flip_sign = -1
            else:
                flip_sign = 1

            # Forward pass
            score = self.forward(white_features, black_features)

            if flip_sign == -1:
                score = -score

            return score.item()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = NNUEModel()
    print(f"NNUE Model Architecture:")
    print(f"Input size: {model.input_size:,}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"\nModel structure:")
    print(model)

    # Test with a starting position
    board = chess.Board()
    white_idx, black_idx = HalfKP.board_to_features(board)
    print(f"\nStarting position features:")
    print(f"White features: {len(white_idx)} active features")
    print(f"Black features: {len(black_idx)} active features")

    # Test forward pass
    white_features = torch.zeros(1, HalfKP.FEATURE_SIZE)
    black_features = torch.zeros(1, HalfKP.FEATURE_SIZE)
    white_features[0, white_idx] = 1.0
    black_features[0, black_idx] = 1.0

    output = model(white_features, black_features)
    print(f"Model output: {output.item():.2f} centipawns")
