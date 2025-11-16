"""
Dataset loader for NNUE training
Supports loading chess positions with evaluations from JSONL format
Includes sparse encoding support for efficient memory usage
"""

import os
import json
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import chess
from .model import HalfKP
from .config import TrainingConfig


class ChessPositionDataset(Dataset):
    """
    PyTorch Dataset for chess positions with evaluations

    Expected data format (JSONL - one JSON object per line):
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "eval": 0.15}
        {"fen": "...", "eval": -0.5}
        ...
    """

    def __init__(self, data_path: str, max_positions: Optional[int] = None,
                 score_min: Optional[float] = None, score_max: Optional[float] = None):
        """Initialize dataset"""
        self.data_path = data_path
        self.positions = []
        self.evaluations = []
        self.score_min = score_min
        self.score_max = score_max

        # Load JSONL data
        if not data_path.endswith('.jsonl'):
            raise ValueError(f"Only JSONL format is supported. Got: {data_path}")
        
        self._load_jsonl(data_path, max_positions)
        print(f"Loaded {len(self.positions)} positions from {data_path}")

    def _load_jsonl(self, path, max_positions):
        """Load positions from JSONL file (one JSON per line)"""
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_positions and i >= max_positions:
                    break

                item = json.loads(line)
                self.positions.append(item['fen'])
                eval_score = float(item['eval'])
                
                # Clip score if clipping parameters are provided
                if self.score_min is not None or self.score_max is not None:
                    if self.score_min is not None:
                        eval_score = max(self.score_min, eval_score)
                    if self.score_max is not None:
                        eval_score = min(self.score_max, eval_score)
                
                self.evaluations.append(eval_score)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample

        Returns:
            Tuple of (white_features, black_features, evaluation)
        """
        fen = self.positions[idx]
        eval_score = self.evaluations[idx]

        # Parse FEN to board
        board = chess.Board(fen)

        # Get HalfKP features
        white_idx, black_idx = HalfKP.board_to_features(board)

        # Convert to dense feature vectors
        white_features = torch.zeros(HalfKP.FEATURE_SIZE, dtype=torch.float32)
        black_features = torch.zeros(HalfKP.FEATURE_SIZE, dtype=torch.float32)

        white_features[white_idx] = 1.0
        black_features[black_idx] = 1.0

        # If it's black to move, swap feature perspectives and flip the evaluation
        if board.turn == chess.BLACK:
            white_features, black_features = black_features, white_features
            eval_score = -eval_score

        # Convert evaluation to tensor
        eval_tensor = torch.tensor([eval_score], dtype=torch.float32)

        return white_features, black_features, eval_tensor


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching

    Returns:
        Tuple of batched tensors
    """
    white_features = torch.stack([item[0] for item in batch])
    black_features = torch.stack([item[1] for item in batch])
    evaluations = torch.stack([item[2] for item in batch])

    return white_features, black_features, evaluations


class ChessPositionDatasetSparse(Dataset):
    """
    PyTorch Dataset for chess positions with evaluations using sparse encoding

    Returns sparse indices instead of dense feature vectors for memory efficiency.
    With HalfKP features, typically only ~30 out of 40960 features are active per position,
    resulting in ~0.1% sparsity.
    """

    def __init__(self, data_path: str, max_positions: Optional[int] = None,
                 score_min: Optional[float] = None, score_max: Optional[float] = None):
        """Initialize sparse dataset"""
        self.data_path = data_path
        self.positions = []
        self.evaluations = []
        self.score_min = score_min
        self.score_max = score_max

        # Load JSONL data
        if not data_path.endswith('.jsonl'):
            raise ValueError(f"Only JSONL format is supported. Got: {data_path}")

        self._load_jsonl(data_path, max_positions)
        print(f"Loaded {len(self.positions)} positions from {data_path} (sparse mode)")

    def _load_jsonl(self, path, max_positions):
        """Load positions from JSONL file (one JSON per line)"""
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_positions and i >= max_positions:
                    break

                item = json.loads(line)
                self.positions.append(item['fen'])
                eval_score = float(item['eval'])
                
                # Clip score if clipping parameters are provided
                if self.score_min is not None or self.score_max is not None:
                    if self.score_min is not None:
                        eval_score = max(self.score_min, eval_score)
                    if self.score_max is not None:
                        eval_score = min(self.score_max, eval_score)
                
                self.evaluations.append(eval_score)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], float]:
        """
        Get a training sample with sparse encoding

        Returns:
            Tuple of (white_feature_indices, black_feature_indices, evaluation)
            where indices are lists of active feature indices
        """
        fen = self.positions[idx]
        eval_score = self.evaluations[idx]

        # Parse FEN to board
        board = chess.Board(fen)

        # Get HalfKP features as sparse indices
        white_idx, black_idx = HalfKP.board_to_features(board)

        # If it's black to move, swap feature perspectives and flip the evaluation
        if board.turn == chess.BLACK:
            white_idx, black_idx = black_idx, white_idx
            eval_score = -eval_score

        return white_idx, black_idx, eval_score


def collate_sparse(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for sparse batching

    Creates 2D sparse tensors where dimensions are (batch_index, feature_index).
    This is much more memory efficient than dense tensors for NNUE features.

    Returns:
        Tuple of (white_indices, white_values, black_indices, black_values, evaluations)
        where indices are [2, nnz] tensors with (batch_idx, feature_idx) coordinates
    """
    white_indices_list = []
    black_indices_list = []
    evaluations = []

    for batch_idx, (white_idx, black_idx, eval_score) in enumerate(batch):
        # Create 2D coordinates: (batch_index, feature_index)
        batch_indices = torch.full((len(white_idx),), batch_idx, dtype=torch.long)
        white_feature_indices = torch.tensor(white_idx, dtype=torch.long)
        white_coords = torch.stack([batch_indices, white_feature_indices], dim=0)
        white_indices_list.append(white_coords)

        batch_indices = torch.full((len(black_idx),), batch_idx, dtype=torch.long)
        black_feature_indices = torch.tensor(black_idx, dtype=torch.long)
        black_coords = torch.stack([batch_indices, black_feature_indices], dim=0)
        black_indices_list.append(black_coords)

        evaluations.append(eval_score)

    # Concatenate all sparse coordinates
    white_indices = torch.cat(white_indices_list, dim=1)  # [2, total_nnz]
    black_indices = torch.cat(black_indices_list, dim=1)  # [2, total_nnz]

    # All values are 1.0 for binary features
    white_values = torch.ones(white_indices.shape[1], dtype=torch.float32)
    black_values = torch.ones(black_indices.shape[1], dtype=torch.float32)

    # Stack evaluations
    evaluations = torch.tensor(evaluations, dtype=torch.float32).unsqueeze(1)

    return white_indices, white_values, black_indices, black_values, evaluations


def create_dataloaders(train_path: str, val_path: Optional[str] = None, 
                       batch_size: int = 256, num_workers: int = 4,
                       max_train_positions: Optional[int] = None, 
                       max_val_positions: Optional[int] = None,
                       score_min: Optional[float] = None,
                       score_max: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_path not provided
    """
    # Create training dataset
    train_dataset = ChessPositionDataset(train_path, max_positions=max_train_positions,
                                         score_min=score_min, score_max=score_max)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create validation dataset if path provided
    val_loader = None
    if val_path:
        val_dataset = ChessPositionDataset(val_path, max_positions=max_val_positions,
                                           score_min=score_min, score_max=score_max)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return train_loader, val_loader


def create_dataloaders_sparse(train_path: str, val_path: Optional[str] = None,
                               batch_size: int = 256, num_workers: int = 0,
                               max_train_positions: Optional[int] = None,
                               max_val_positions: Optional[int] = None,
                               score_min: Optional[float] = None,
                               score_max: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders with sparse encoding

    Note: num_workers must be 0 for sparse tensors to avoid multiprocessing issues.
    Sparse encoding is still much faster due to reduced memory usage and computation.

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_path not provided
    """
    if num_workers > 0:
        print("Warning: num_workers > 0 with sparse encoding. Setting to 0 to avoid multiprocessing issues.")
        num_workers = 0

    # Create training dataset
    train_dataset = ChessPositionDatasetSparse(train_path, max_positions=max_train_positions,
                                               score_min=score_min, score_max=score_max)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_sparse,
        pin_memory=True if num_workers == 0 else False
    )

    # Create validation dataset if path provided
    val_loader = None
    if val_path:
        val_dataset = ChessPositionDatasetSparse(val_path, max_positions=max_val_positions,
                                                  score_min=score_min, score_max=score_max)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sparse,
            pin_memory=True if num_workers == 0 else False
        )

    return train_loader, val_loader


def split_dataset(data_path: str, train_ratio: Optional[float] = None, 
                  output_dir: Optional[str] = None, 
                  config: Optional[TrainingConfig] = None) -> Tuple[str, str]:
    """
    Split a dataset into train and validation sets

    Returns:
        Tuple of (train_path, val_path)
    """
    import numpy as np

    # Use config defaults if not provided
    if train_ratio is None:
        train_ratio = config.train_val_split_ratio if config else 0.9
    if output_dir is None:
        output_dir = config.download_output_dir if config else 'data'

    os.makedirs(output_dir, exist_ok=True)

    # Load all positions (with clipping if config provided)
    score_min = config.eval_score_min if config else None
    score_max = config.eval_score_max if config else None
    dataset = ChessPositionDataset(data_path, score_min=score_min, score_max=score_max)

    # Create indices and shuffle (using numpy for reproducibility with set_seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices = indices.tolist()

    # Split
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Write train and validation data as JSONL
    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')

    with open(train_path, 'w') as f:
        for idx in train_indices:
            json.dump({'fen': dataset.positions[idx], 'eval': dataset.evaluations[idx]}, f)
            f.write('\n')

    with open(val_path, 'w') as f:
        for idx in val_indices:
            json.dump({'fen': dataset.positions[idx], 'eval': dataset.evaluations[idx]}, f)
            f.write('\n')

    print(f"Split dataset:")
    print(f"  Train: {len(train_indices)} positions -> {train_path}")
    print(f"  Val:   {len(val_indices)} positions -> {val_path}")

    return train_path, val_path
