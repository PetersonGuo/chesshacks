"""
Configuration for NNUE training
"""

import os
import copy
from dataclasses import dataclass
from typing import Optional

# Check device availability once at module level
try:
    import torch
    _cuda_available = torch.cuda.is_available()
    _mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    _cuda_available = False
    _mps_available = False

# Track if warnings have been printed
_device_warning_printed = False


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration"""

    # Data paths
    train_data_path: str = 'data/train.jsonl'
    val_data_path: str = 'data/val.jsonl'

    # Training parameters
    batch_size: int = 512  # Increased for better GPU utilization
    learning_rate: float = 0.0005  # Reduced further to help with overfitting and improve generalization
    num_epochs: int = 30
    weight_decay: float = 1e-4  # Increased from 1e-5 to help with overfitting
    max_grad_norm: Optional[float] = 1.0  # Gradient clipping to prevent explosion (None = disabled)
    early_stopping_patience: Optional[int] = 15  # Stop if val loss doesn't improve for N epochs (None = disabled) - increased to allow more exploration

    # Model architecture
    hidden_size: int = 256
    hidden2_size: int = 32
    hidden3_size: int = 32
    
    # CNN model architecture (for CNN-based training)
    conv_channels: int = 64
    num_residual_blocks: int = 3  # Reduced from 4 to help with overfitting
    dense_hidden: int = 256
    dropout: float = 0.5  # Increased from 0.3 to help with overfitting

    # Data loading
    num_workers: int = 16
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
    device: str = 'cuda'  # 'cuda', 'mps', or 'cpu' - defaults to MPS (Apple Silicon GPU)

    # Optimization
    optimizer: str = 'adam'  # 'adam' or 'sgd'
    scheduler: Optional[str] = 'cosine'  # 'step', 'cosine', or None - cosine helps with overfitting
    scheduler_step_size: int = 30  # for StepLR
    scheduler_gamma: float = 0.1  # for StepLR
    
    # Warmup
    warmup_epochs: int = 5  # Number of epochs for learning rate warmup (0 = no warmup) - reduced for 50 total epochs
    warmup_start_lr: Optional[float] = 1e-5  # Starting learning rate for warmup (None = 0.0, recommended: 1e-5 to 1e-4)

    # Loss function
    loss_function: str = 'huber'  # 'mse' or 'huber' - huber is more robust to outliers
    
    # Score clipping
    eval_score_min: int = -1500  # Minimum evaluation score in centipawns (clipped) - reduced from -10000 to focus on typical positions
    eval_score_max: int = 1500  # Maximum evaluation score in centipawns (clipped) - reduced from 10000 to focus on typical positions

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
    download_max_games: Optional[int] = 100000  # Maximum games to download (simplified workflow)
    download_max_games_searched: Optional[int] = None  # Maximum total games to search (None = unlimited)
    download_depth: int = 10
    download_positions_per_game: int = 20  # Increased to extract more positions per game
    download_num_workers: int = 16
    download_batch_size: int = 100
    download_rated_only: bool = True  # Download rated games only
    download_output_dir: str = 'data'  # Output directory for downloaded data
    download_mode: str = 'streaming'  # Download mode: 'streaming' or 'direct'
    download_skip_filter: bool = True  # Skip filtering entirely - use all games from downloaded database (simplified workflow)
    download_skip_redownload: bool = True  # Skip re-downloading if file already exists
    
    resume_from: Optional[str] = None

    # Hugging Face upload configuration
    hf_repo_id: str = 'jleezhang/chesshacks_model'
    hf_auto_upload: bool = False  # Automatically upload checkpoints and final model (set to True to enable)
    hf_upload_all_checkpoints: bool = False  # Only upload current checkpoint, not all previous ones
    hf_checkpoints_dir: Optional[str] = None
    hf_model_name: Optional[str] = None
    hf_model_description: Optional[str] = None
    hf_private: bool = False

    def __post_init__(self):
        """Validate and create necessary directories"""
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Validate device (only print warning once)
        global _device_warning_printed
        
        # Check flag first - if already printed, silently set device without printing
        if _device_warning_printed:
            if self.device == 'cuda' and not _cuda_available:
                self.device = 'mps' if _mps_available else 'cpu'
            elif self.device == 'mps' and not _mps_available:
                self.device = 'cpu'
            return
        
        # First time through - silently set device without printing
        if self.device == 'cuda' and not _cuda_available:
            # Try MPS first, then CPU
            if _mps_available:
                _device_warning_printed = True
                self.device = 'mps'
            else:
                _device_warning_printed = True
                self.device = 'cpu'
        elif self.device == 'mps' and not _mps_available:
            _device_warning_printed = True
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

# RTX 5070 optimized configuration
RTX_5070_CONFIG = TrainingConfig(
    # Device
    device='cuda',

    # Batch size optimized for RTX 5070 (12-16GB VRAM)
    # With bitmap model (768 features), we can use very large batches
    batch_size=4096,  # Large batch for efficient GPU utilization

    # Data loading - optimize for fast PCIe 4.0+ transfer
    num_workers=8,  # Balanced for CPU->GPU pipeline

    # Training parameters
    learning_rate=0.001,  # Higher LR works well with large batches
    num_epochs=100,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    early_stopping_patience=20,

    # Model architecture - balanced for speed and quality
    hidden_size=256,
    hidden2_size=32,
    hidden3_size=32,

    # Optimization
    optimizer='adam',
    scheduler='cosine',
    warmup_epochs=5,
    warmup_start_lr=1e-5,

    # Loss function
    loss_function='huber',

    # Score clipping
    eval_score_min=-1500,
    eval_score_max=1500,

    # Checkpointing
    save_every_n_epochs=5,
    save_best_only=False,
    validate_every_n_epochs=1,

    # Data - None = use all available data
    max_train_positions=None,
    max_val_positions=None,

    # Logging
    log_every_n_batches=50,
    verbose=True,
)

# RTX 5070 high-quality configuration (larger model)
RTX_5070_QUALITY_CONFIG = TrainingConfig(
    # Device
    device='cuda',

    # Smaller batch for larger model
    batch_size=2048,

    # Data loading
    num_workers=8,

    # Training parameters
    learning_rate=0.0005,  # Lower LR for stability with larger model
    num_epochs=150,
    weight_decay=1e-5,
    max_grad_norm=1.0,
    early_stopping_patience=25,

    # Larger model architecture
    hidden_size=512,  # Doubled from default
    hidden2_size=64,  # Doubled from default
    hidden3_size=64,  # Doubled from default

    # Optimization
    optimizer='adam',
    scheduler='cosine',
    warmup_epochs=10,
    warmup_start_lr=1e-6,

    # Loss function
    loss_function='huber',

    # Score clipping
    eval_score_min=-1500,
    eval_score_max=1500,

    # Checkpointing
    save_every_n_epochs=5,
    save_best_only=False,
    validate_every_n_epochs=1,

    # Data
    max_train_positions=None,
    max_val_positions=None,

    # Logging
    log_every_n_batches=50,
    verbose=True,
)


def get_config(config_name='default'):
    """
    Get a configuration by name

    Args:
        config_name: One of 'default', 'fast', 'quality', 'large_scale', 'rtx5070', 'rtx5070_quality'

    Returns:
        TrainingConfig instance
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'fast': FAST_CONFIG,
        'quality': QUALITY_CONFIG,
        'large_scale': LARGE_SCALE_CONFIG,
        'rtx5070': RTX_5070_CONFIG,
        'rtx5070_quality': RTX_5070_QUALITY_CONFIG,
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
