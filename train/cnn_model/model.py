"""
CNN-based Chess Evaluation Model
Uses convolutional layers to learn spatial patterns on the chess board.
More data-efficient than NNUE and doesn't require hand-crafted features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessCNNModel(nn.Module):
    """
    CNN-based chess evaluation model
    
    Architecture:
    - Input: 8x8x12 board representation (12 channels for piece types/colors)
    - Convolutional layers to extract spatial features
    - Residual blocks for deeper learning
    - Dense layers for final evaluation
    
    This architecture is more data-efficient than NNUE and can learn
    spatial patterns directly from the board representation.
    """
    
    def __init__(self, 
                 input_channels=12,
                 conv_channels=64,
                 num_residual_blocks=4,
                 dense_hidden=256,
                 dropout=0.3):
        super(ChessCNNModel, self).__init__()
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(conv_channels) for _ in range(num_residual_blocks)
        ])
        
        # Additional conv layer after residuals
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels * 2)
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(conv_channels * 2, dense_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, dense_hidden // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(dense_hidden // 2, 1)
        
        # Output scaling factor - with normalization, we use a smaller scale
        # tanh outputs [-1, 1], so we'll scale to reasonable normalized range
        # Since normalized targets have std ~1, we use scale of 3 to cover ±3 std
        self.output_scale = 3.0
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m == self.fc3:
                    # Initialize final layer with small weights to output reasonable values
                    # This helps the model start in a reasonable range
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Board representation [batch, 12, 8, 8]
               Channels: [white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
                         black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king]
        
        Returns:
            Evaluation score in centipawns [batch, 1]
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Additional convolution
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Global average pooling: [batch, channels*2, 8, 8] -> [batch, channels*2, 1, 1]
        x = self.global_pool(x)
        
        # Flatten: [batch, channels*2, 1, 1] -> [batch, channels*2]
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Scale output using tanh to bound values
        # tanh outputs [-1, 1], so multiply by output_scale
        # With normalized targets (std ~1), scale of 3 covers ±3 standard deviations
        x = torch.tanh(x) * self.output_scale
        
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
            
            # Convert board to tensor
            board_tensor = board_to_tensor(board, device=device, dtype=dtype)
            
            # Forward pass
            score = self.forward(board_tensor)
            
            return score.item()


def board_to_tensor(board: chess.Board, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Convert a chess board to a tensor representation
    
    Args:
        board: python-chess Board object
        device: torch device
        dtype: torch dtype
    
    Returns:
        Tensor of shape [1, 12, 8, 8] representing the board
        Channels: [white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
                  black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king]
    """
    # Create 8x8 board representation
    board_array = torch.zeros(12, 8, 8, dtype=dtype, device=device)
    
    # Piece type to channel mapping
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    # Fill board representation
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = 7 - (square // 8)  # Convert to row-major (0,0 is top-left)
            col = square % 8
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            board_array[channel, row, col] = 1.0
    
    # Add batch dimension: [12, 8, 8] -> [1, 12, 8, 8]
    board_tensor = board_array.unsqueeze(0)
    
    # If it's black to move, flip the board (mirror vertically and swap colors)
    if board.turn == chess.BLACK:
        # Flip board vertically (mirror along horizontal axis)
        board_tensor = torch.flip(board_tensor, dims=[2])
        # Swap white and black pieces (more efficient than clone)
        board_tensor = torch.cat([
            board_tensor[:, 6:, :, :],  # black pieces
            board_tensor[:, :6, :, :]   # white pieces
        ], dim=1)
    
    return board_tensor


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ChessCNNModel(
        input_channels=12,
        conv_channels=64,
        num_residual_blocks=4,
        dense_hidden=256,
        dropout=0.3
    )
    
    print(f"CNN Model Architecture:")
    print(f"Input: 8x8x12 board representation")
    print(f"Convolutional channels: 64")
    print(f"Residual blocks: 4")
    print(f"Dense hidden: 256")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"\nModel structure:")
    print(model)
    
    # Test with a starting position
    board = chess.Board()
    board_tensor = board_to_tensor(board)
    print(f"\nStarting position tensor shape: {board_tensor.shape}")
    
    # Test forward pass
    output = model(board_tensor)
    print(f"Model output: {output.item():.2f} centipawns")
    
    # Test evaluation
    eval_score = model.evaluate_board(board)
    print(f"Board evaluation: {eval_score:.2f} centipawns")

