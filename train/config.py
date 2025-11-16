"""
Configuration for NNUE training
"""

import os
import copy
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration"""

    # Data paths
    train_data_path: str = 'data/train.jsonl'
    val_data_path: str = 'data/val.jsonl'

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.003  # Higher LR for faster initial learning (was 0.001)
    num_epochs: int = 300
    weight_decay: float = 1e-5
    max_grad_norm: Optional[float] = 1.0  # Gradient clipping to prevent explosion (None = disabled)

    # Model architecture
    hidden_size: int = 256
    hidden2_size: int = 32
    hidden3_size: int = 32

    # Data loading
    num_workers: int = 4
    max_train_positions: Optional[int] = None  # None = load all
    max_val_positions: Optional[int] = None
    train_val_split_ratio: float = 0.9  # Ratio of data to use for training (rest for validation)

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 10
    save_best_only: bool = False

    # Validation
    validate_every_n_epochs: int = 1

    # Device
    device: str = 'cuda'  # 'cuda', 'mps', or 'cpu'

    # Optimization
    optimizer: str = 'adam'  # 'adam' or 'sgd'
    scheduler: Optional[str] = None  # 'step', 'cosine', or None - set to None if you want LR to stay constant after warmup
    scheduler_step_size: int = 30  # for StepLR
    scheduler_gamma: float = 0.1  # for StepLR
    
    # Warmup
    warmup_epochs: int = 100  # Number of epochs for learning rate warmup (0 = no warmup)
    warmup_start_lr: Optional[float] = 1e-5  # Starting learning rate for warmup (None = 0.0, recommended: 1e-5 to 1e-4)

    # Loss function
    loss_function: str = 'mse'  # 'mse' or 'huber'

    # Logging
    log_every_n_batches: int = 100
    verbose: bool = True

    # Random seed
    seed: int = 42

    # Data download
    auto_download: bool = True
    stockfish_path: str = 'stockfish'
    download_year: Optional[int] = None  # Use latest month (None = auto-detect latest)
    download_month: Optional[int] = None  # Use latest month (None = auto-detect latest)
    download_max_games: Optional[int] = 10000  # Maximum games to download (simplified workflow)
    download_max_games_searched: Optional[int] = None  # Maximum total games to search (None = unlimited)
    download_depth: int = 10
    download_positions_per_game: int = 10
    download_num_workers: int = 4
    download_batch_size: int = 100
    download_rated_only: bool = True  # Download rated games only
    download_output_dir: str = 'data'  # Output directory for downloaded data
    download_mode: str = 'streaming'  # Download mode: 'streaming' or 'direct'
    download_skip_filter: bool = True  # Skip filtering entirely - use all games from downloaded database (simplified workflow)
    download_skip_redownload: bool = True  # Skip re-downloading if file already exists
    
    resume_from: Optional[str] = None

    # Hugging Face upload configuration
    hf_repo_id: str = 'jleezhang/chesshacks_model'
    hf_auto_upload: bool = True  # Automatically upload checkpoints and final model
    hf_upload_all_checkpoints: bool = False  # Only upload current checkpoint, not all previous ones
    hf_checkpoints_dir: Optional[str] = None
    hf_model_name: Optional[str] = None
    hf_model_description: Optional[str] = None
    hf_private: bool = False

    def __post_init__(self):
        """Validate and create necessary directories"""
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Validate device
        import torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        elif self.device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = 'cpu'


# Default configuration
DEFAULT_CONFIG = TrainingConfig()


# Example configurations for different scenarios

# Fast training for testing
FAST_CONFIG = TrainingConfig(
    batch_size=512,
    num_epochs=10,
    max_train_positions=10000,
    max_val_positions=1000,
    save_every_n_epochs=2,
)

# High-quality training
QUALITY_CONFIG = TrainingConfig(
    batch_size=256,
    learning_rate=0.0005,
    num_epochs=200,
    hidden_size=512,
    hidden2_size=64,
    hidden3_size=64,
    weight_decay=1e-6,
    optimizer='adam',
    scheduler='cosine',
)

# Large-scale training
LARGE_SCALE_CONFIG = TrainingConfig(
    batch_size=1024,
    learning_rate=0.001,
    num_epochs=100,
    hidden_size=256,
    num_workers=8,
    save_every_n_epochs=10,
)


def get_config(config_name='default'):
    """
    Get a configuration by name

    Args:
        config_name: One of 'default', 'fast', 'quality', 'large_scale'

    Returns:
        TrainingConfig instance
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'fast': FAST_CONFIG,
        'quality': QUALITY_CONFIG,
        'large_scale': LARGE_SCALE_CONFIG,
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")

    config = copy.deepcopy(configs[config_name])
    return config


if __name__ == "__main__":
    # Print default configuration
    print("Default Configuration:")
    print("-" * 50)
    config = DEFAULT_CONFIG
    for field, value in config.__dict__.items():
        print(f"{field:30s}: {value}")

    print("\n\nFast Configuration (for testing):")
    print("-" * 50)
    fast_config = FAST_CONFIG
    for field, value in fast_config.__dict__.items():
        print(f"{field:30s}: {value}")
