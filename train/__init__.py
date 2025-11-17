"""
NNUE training package – requires PyTorch and related dependencies.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the training package. Please install torch first."
    ) from exc

from .config import TrainingConfig, get_config
from .dataset import ChessPositionDataset, create_dataloaders, split_dataset
from .download_data import download_and_process_lichess_data
from .model import HalfKP, NNUEModel, count_parameters
from .train import train
from .upload_to_hf import upload_model_to_hf

__all__ = [
    "NNUEModel",
    "HalfKP",
    "count_parameters",
    "TrainingConfig",
    "get_config",
    "ChessPositionDataset",
    "create_dataloaders",
    "split_dataset",
    "train",
    "download_and_process_lichess_data",
    "upload_model_to_hf",
]
