"""
Dataset loader for NNUE training
Supports loading chess positions with evaluations from various formats
"""

import os
import json
import csv
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import chess
from .model import HalfKP
from .config import TrainingConfig


class ChessPositionDataset(Dataset):
    """
    PyTorch Dataset for chess positions with evaluations

    Expected data format (CSV):
        fen,eval
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0.15
        ...

    Expected data format (JSON):
        [
            {"fen": "rnbqkbnr/...", "eval": 0.15},
            ...
        ]

    Or JSONL (one JSON object per line):
        {"fen": "rnbqkbnr/...", "eval": 0.15}
        {"fen": "...", "eval": -0.5}
    """

    def __init__(self, data_path: str, max_positions: Optional[int] = None):
        """
        Initialize dataset

        Args:
            data_path: Path to data file (CSV, JSON, or JSONL)
            max_positions: Maximum number of positions to load (None = load all)
        """
        self.data_path = data_path
        self.positions = []
        self.evaluations = []

        # Load data based on file extension
        if data_path.endswith('.csv'):
            self._load_csv(data_path, max_positions)
        elif data_path.endswith('.json'):
            self._load_json(data_path, max_positions)
        elif data_path.endswith('.jsonl'):
            self._load_jsonl(data_path, max_positions)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        print(f"Loaded {len(self.positions)} positions from {data_path}")

    def _load_csv(self, path, max_positions):
        """Load positions from CSV file"""
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_positions and i >= max_positions:
                    break

                fen = row['fen']
                eval_score = float(row['eval'])

                self.positions.append(fen)
                self.evaluations.append(eval_score)

    def _load_json(self, path, max_positions):
        """Load positions from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            if max_positions and i >= max_positions:
                break

            self.positions.append(item['fen'])
            self.evaluations.append(float(item['eval']))

    def _load_jsonl(self, path, max_positions):
        """Load positions from JSONL file (one JSON per line)"""
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_positions and i >= max_positions:
                    break

                item = json.loads(line)
                self.positions.append(item['fen'])
                self.evaluations.append(float(item['eval']))

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

    Args:
        batch: List of (white_features, black_features, eval) tuples

    Returns:
        Tuple of batched tensors
    """
    white_features = torch.stack([item[0] for item in batch])
    black_features = torch.stack([item[1] for item in batch])
    evaluations = torch.stack([item[2] for item in batch])

    return white_features, black_features, evaluations


def create_dataloaders(train_path: str, val_path: Optional[str] = None, 
                       batch_size: int = 256, num_workers: int = 4,
                       max_train_positions: Optional[int] = None, 
                       max_val_positions: Optional[int] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders

    Args:
        train_path: Path to training data file
        val_path: Path to validation data file (optional)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        max_train_positions: Maximum training positions to load
        max_val_positions: Maximum validation positions to load

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_path not provided
    """
    # Create training dataset
    train_dataset = ChessPositionDataset(train_path, max_positions=max_train_positions)
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
        val_dataset = ChessPositionDataset(val_path, max_positions=max_val_positions)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return train_loader, val_loader


def split_dataset(data_path: str, train_ratio: Optional[float] = None, 
                  output_dir: Optional[str] = None, 
                  config: Optional[TrainingConfig] = None) -> Tuple[str, str]:
    """
    Split a dataset into train and validation sets

    Args:
        data_path: Path to original data file
        train_ratio: Ratio of data to use for training (defaults to config.train_val_split_ratio or 0.9)
        output_dir: Directory to save split files (defaults to config.download_output_dir or 'data')
        config: TrainingConfig instance (optional, for defaults)

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

    # Load all positions
    dataset = ChessPositionDataset(data_path)

    # Create indices and shuffle (using numpy for reproducibility with set_seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices = indices.tolist()

    # Split
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Determine output format from input
    ext = os.path.splitext(data_path)[1]
    train_path = os.path.join(output_dir, f'train{ext}')
    val_path = os.path.join(output_dir, f'val{ext}')

    # Write train data
    if ext == '.csv':
        with open(train_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fen', 'eval'])
            for idx in train_indices:
                writer.writerow([dataset.positions[idx], dataset.evaluations[idx]])

        with open(val_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fen', 'eval'])
            for idx in val_indices:
                writer.writerow([dataset.positions[idx], dataset.evaluations[idx]])

    elif ext == '.json':
        train_records = [
            {'fen': dataset.positions[idx], 'eval': dataset.evaluations[idx]}
            for idx in train_indices
        ]
        val_records = [
            {'fen': dataset.positions[idx], 'eval': dataset.evaluations[idx]}
            for idx in val_indices
        ]

        with open(train_path, 'w') as f:
            json.dump(train_records, f)

        with open(val_path, 'w') as f:
            json.dump(val_records, f)

    elif ext == '.jsonl':
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
