"""
NNUE (Efficiently Updatable Neural Network) Model Architecture
HalfKP variant for chess position evaluation
"""

import torch
import torch.nn as nn
import chess
from typing import Optional


class ClippedReLU(nn.Module):
    """Clipped ReLU activation: min(max(x, 0), 1)"""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class SparseLinear(nn.Module):
    """
    Efficient sparse linear layer for NNUE feature transformer

    Performs linear transformation on sparse inputs by only accessing
    weights corresponding to active features, avoiding dense matrix multiplication.
    """

    def __init__(self, in_features: int, out_features: int):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix: [in_features, out_features]
        # Transposed from standard nn.Linear for efficient indexing
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.weight, gain=0.5)
        nn.init.constant_(self.bias, 0)

    def forward(self, indices: torch.Tensor, values: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Forward pass with sparse inputs

        Args:
            indices: [2, nnz] tensor with (batch_idx, feature_idx) coordinates
            values: [nnz] tensor with feature values (typically all 1.0)
            batch_size: Number of samples in the batch

        Returns:
            Dense output tensor [batch_size, out_features]
        """
        # Extract batch and feature indices
        batch_idx = indices[0]  # [nnz]
        feature_idx = indices[1]  # [nnz]

        # Gather weights for active features: [nnz, out_features]
        active_weights = self.weight[feature_idx]  # [nnz, out_features]

        # Multiply by values (element-wise): [nnz, out_features]
        weighted = active_weights * values.unsqueeze(1)  # [nnz, out_features]

        # Scatter-add to output tensor
        output = torch.zeros(batch_size, self.out_features,
                           dtype=weighted.dtype, device=weighted.device)
        output.index_add_(0, batch_idx, weighted)

        # Add bias
        output = output + self.bias

        return output


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


class NNUEModelSparse(nn.Module):
    """
    NNUE Neural Network with sparse encoding for chess position evaluation

    Uses sparse linear layers for the feature transformer, which is much more
    efficient given the sparsity of HalfKP features (~0.1% active features).

    Architecture: HalfKP features (sparse) → 256 → 32 → 32 → 1
    """

    def __init__(self, hidden_size=256, hidden2_size=32, hidden3_size=32):
        super(NNUEModelSparse, self).__init__()

        self.input_size = HalfKP.FEATURE_SIZE
        self.hidden_size = hidden_size

        # Sparse feature transformer: converts sparse HalfKP features to dense representation
        # We have two perspectives (white and black), each gets transformed separately
        self.ft = SparseLinear(self.input_size, hidden_size)

        # Hidden layers (dense)
        self.fc1 = nn.Linear(hidden_size * 2, hidden2_size)  # *2 because we concat white+black
        self.fc2 = nn.Linear(hidden2_size, hidden3_size)
        self.fc3 = nn.Linear(hidden3_size, 1)

        # Activations
        self.crelu = ClippedReLU()

        # Initialize dense layer weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, white_indices, white_values, black_indices, black_values, batch_size: Optional[int] = None):
        """
        Forward pass through the network with sparse inputs

        Args:
            white_indices: [2, nnz_w] tensor with (batch_idx, feature_idx) for white
            white_values: [nnz_w] tensor with feature values for white
            black_indices: [2, nnz_b] tensor with (batch_idx, feature_idx) for black
            black_values: [nnz_b] tensor with feature values for black

        Returns:
            Position evaluation in centipawns [batch_size, 1]
        """
        # Infer batch size from indices if not provided
        if batch_size is None:
            inferred_batch = -1
            if white_indices.numel() > 0:
                inferred_batch = max(inferred_batch, int(white_indices[0].max().item()))
            if black_indices.numel() > 0:
                inferred_batch = max(inferred_batch, int(black_indices[0].max().item()))
            if inferred_batch < 0:
                raise ValueError(
                    "Unable to infer batch size from empty sparse indices. "
                    "Please provide batch_size."
                )
            batch_size = inferred_batch + 1

        # Transform features from both perspectives using sparse operations
        w = self.crelu(self.ft(white_indices, white_values, batch_size))
        b = self.crelu(self.ft(black_indices, black_values, batch_size))

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

            # Get sparse features
            white_idx, black_idx = HalfKP.board_to_features(board)

            # Create sparse tensors for single position (batch_size=1)
            white_indices = torch.tensor([[0] * len(white_idx), white_idx],
                                       dtype=torch.long, device=device)
            white_values = torch.ones(len(white_idx), dtype=torch.float32, device=device)

            black_indices = torch.tensor([[0] * len(black_idx), black_idx],
                                       dtype=torch.long, device=device)
            black_values = torch.ones(len(black_idx), dtype=torch.float32, device=device)

            if board.turn == chess.BLACK:
                white_indices, black_indices = black_indices, white_indices
                white_values, black_values = black_values, white_values
                flip_sign = -1
            else:
                flip_sign = 1

            # Forward pass
            score = self.forward(white_indices, white_values, black_indices, black_values, batch_size=1)

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
