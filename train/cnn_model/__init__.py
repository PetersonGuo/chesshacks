"""
CNN-based chess evaluation model
"""

from .model import ChessCNNModel, board_to_tensor, count_parameters

__all__ = ['ChessCNNModel', 'board_to_tensor', 'count_parameters']

