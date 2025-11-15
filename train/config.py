"""
Configuration for NNUE training
"""

import os
import copy
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration"""

    # Data paths
    train_data_path: str = 'data/train.csv'
    val_data_path: str = 'data/val.csv'

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-5

    # Model architecture
    hidden_size: int = 256
    hidden2_size: int = 32
    hidden3_size: int = 32

    # Data loading
    num_workers: int = 4
    max_train_positions: int = None  # None = load all
    max_val_positions: int = None

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 5
    save_best_only: bool = True

    # Validation
    validate_every_n_epochs: int = 1

    # Device
    device: str = 'cuda'  # 'cuda', 'mps', or 'cpu'

    # Optimization
    optimizer: str = 'adam'  # 'adam' or 'sgd'
    scheduler: str = None  # 'step', 'cosine', or None
    scheduler_step_size: int = 30  # for StepLR
    scheduler_gamma: float = 0.1  # for StepLR

    # Loss function
    loss_function: str = 'mse'  # 'mse' or 'huber'

    # Logging
    log_every_n_batches: int = 100
    verbose: bool = True

    # Random seed
    seed: int = 42

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

    return copy.deepcopy(configs[config_name])


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
