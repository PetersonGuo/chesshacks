"""
Main training script for NNUE-based chess evaluation model
Uses HalfKP features with sparse encoding for fast inference.
NNUE is optimized for efficient incremental updates during search.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import threading
from typing import List, Optional, Tuple, Dict
import json
import chess
from torch.utils.data import Dataset, DataLoader

# Handle both direct execution and package import
try:
    from .model import ChessNNUEModel, HalfKP, count_parameters
    from ..config import get_config, TrainingConfig
    from ..download_data import download_and_process_lichess_data
    from ..upload_to_hf import upload_model_to_hf
    from ..dataset import ChessPositionDatasetSparse, collate_sparse, create_dataloaders_sparse
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from train.nnue_model.model import ChessNNUEModel, HalfKP, count_parameters
    from train.config import get_config, TrainingConfig
    from train.download_data import download_and_process_lichess_data
    from train.upload_to_hf import upload_model_to_hf
    from train.dataset import ChessPositionDatasetSparse, collate_sparse, create_dataloaders_sparse


class ChessBoardDatasetSparse(Dataset):
    """
    PyTorch Dataset for chess positions with evaluations using sparse NNUE features
    Includes normalization support
    
    Expected data format (JSONL - one JSON object per line):
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "eval": 0.15}
        {"fen": "...", "eval": -0.5}
        ...
    """

    def __init__(self, data_path: str, max_positions: Optional[int] = None,
                 score_min: Optional[float] = None, score_max: Optional[float] = None,
                 normalize: bool = True):
        """Initialize sparse dataset"""
        self.data_path = data_path
        self.positions = []
        self.evaluations = []
        self.score_min = score_min
        self.score_max = score_max
        self.normalize = normalize
        self.eval_mean = 0.0
        self.eval_std = 1.0

        # Load JSONL data
        if not data_path.endswith('.jsonl'):
            raise ValueError(f"Only JSONL format is supported. Got: {data_path}")

        self._load_jsonl(data_path, max_positions)
        
        # Compute normalization statistics if enabled
        if self.normalize and len(self.evaluations) > 0:
            eval_array = np.array(self.evaluations)
            self.eval_mean = float(np.mean(eval_array))
            self.eval_std = float(np.std(eval_array))
            if self.eval_std < 1e-6:  # Avoid division by zero
                self.eval_std = 1.0
            print(f"Loaded {len(self.positions)} positions from {data_path} (sparse mode)")
            print(f"Evaluation stats: mean={self.eval_mean:.2f}, std={self.eval_std:.2f}")
        else:
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

        # Normalize evaluation if enabled
        if self.normalize:
            eval_score = (eval_score - self.eval_mean) / self.eval_std

        return white_idx, black_idx, eval_score


def collate_fn_sparse_normalized(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for sparse batching with normalized evaluations

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
                       batch_size: int = 256, num_workers: int = 0,
                       max_train_positions: Optional[int] = None, 
                       max_val_positions: Optional[int] = None,
                       score_min: Optional[float] = None,
                       score_max: Optional[float] = None,
                       device: Optional[torch.device] = None,
                       normalize: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
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
    train_dataset = ChessBoardDatasetSparse(train_path, max_positions=max_train_positions,
                                           score_min=score_min, score_max=score_max,
                                           normalize=normalize)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_sparse_normalized,
        pin_memory=False  # Sparse tensors don't work well with pin_memory
    )

    # Create validation dataset if path provided
    val_loader = None
    if val_path:
        # Use same normalization stats from training set
        val_dataset = ChessBoardDatasetSparse(val_path, max_positions=max_val_positions,
                                             score_min=score_min, score_max=score_max,
                                             normalize=False)  # Don't recompute stats
        if normalize and hasattr(train_dataset, 'eval_mean'):
            val_dataset.eval_mean = train_dataset.eval_mean
            val_dataset.eval_std = train_dataset.eval_std
            val_dataset.normalize = True
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_sparse_normalized,
            pin_memory=False
        )

    return train_loader, val_loader


class WarmupScheduler:
    """
    Wraps a learning rate scheduler with warmup functionality.
    During warmup epochs, linearly increases learning rate from warmup_start_lr to target_lr.
    After warmup, delegates to the wrapped scheduler.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                 warmup_epochs: int,
                 target_lr: float,
                 warmup_start_lr: Optional[float] = None):
        """
        Args:
            optimizer: The optimizer whose learning rate will be scheduled
            base_scheduler: The scheduler to use after warmup (can be None)
            warmup_epochs: Number of epochs for warmup
            target_lr: Target learning rate after warmup
            warmup_start_lr: Starting learning rate for warmup (None = 0.0)
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else 0.0
        self.current_epoch = -1  # Start at -1 so first step() sets LR for epoch 0

        # Initialize learning rate - will be set properly on first step()
        if warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.warmup_start_lr

    def step(self):
        """Update learning rate for next epoch"""
        self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear interpolation from warmup_start_lr to target_lr
            lr = self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * \
                 (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # After warmup: use base scheduler if available
            if self.base_scheduler is not None:
                self.base_scheduler.step()

    def get_last_lr(self):
        """Get the last computed learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Return state dict for checkpointing"""
        state = {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'target_lr': self.target_lr,
            'warmup_start_lr': self.warmup_start_lr,
        }
        if self.base_scheduler is not None:
            state['base_scheduler'] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.target_lr = state_dict['target_lr']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        if self.base_scheduler is not None and 'base_scheduler' in state_dict:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""
    if config.optimizer == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer: torch.optim.Optimizer,
                  config: TrainingConfig):
    """Create learning rate scheduler based on configuration"""
    base_scheduler = None

    if config.scheduler == 'step':
        base_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler == 'cosine':
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs - config.warmup_epochs
        )

    # Wrap with warmup if enabled
    if config.warmup_epochs > 0:
        return WarmupScheduler(
            optimizer=optimizer,
            base_scheduler=base_scheduler,
            warmup_epochs=config.warmup_epochs,
            target_lr=config.learning_rate,
            warmup_start_lr=config.warmup_start_lr
        )

    return base_scheduler


def get_loss_function(config: TrainingConfig) -> torch.nn.Module:
    """Get loss function based on configuration"""
    if config.loss_function == 'mse':
        return nn.MSELoss()
    elif config.loss_function == 'huber':
        # delta=1.0 works well for normalized targets (std ~1)
        # This makes Huber loss behave like MSE for small errors, L1 for large errors
        return nn.HuberLoss(delta=1.0)
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}")


def train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                device: torch.device, config: TrainingConfig, epoch: int) -> float:
    """
    Train for one epoch

    Args:
        model: NNUE model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        config: Training configuration
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}',
                disable=not config.verbose)

    for batch_idx, (white_indices, white_values, black_indices, black_values, targets) in enumerate(pbar):
        # Move to device
        white_indices = white_indices.to(device)
        white_values = white_values.to(device)
        black_indices = black_indices.to(device)
        black_values = black_values.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(white_indices, white_values, black_indices, black_values)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
        if batch_idx % config.log_every_n_batches == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module, device: torch.device,
             config: TrainingConfig) -> float:
    """
    Validate the model

    Args:
        model: NNUE model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        config: Training configuration

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', disable=not config.verbose)
        for white_indices, white_values, black_indices, black_values, targets in pbar:
            # Move to device
            white_indices = white_indices.to(device)
            white_values = white_values.to(device)
            black_indices = black_indices.to(device)
            black_values = black_values.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(white_indices, white_values, black_indices, black_values)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    config: TrainingConfig,
    filename: str,
    upload_to_hf: bool = False,
    upload_threads: Optional[List[threading.Thread]] = None,
    scheduler = None,
    best_val_loss: Optional[float] = None,
    training_start_time: Optional[str] = None,
) -> None:
    """
    Save model checkpoint

    Args:
        model: NNUE model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss (can be None)
        config: Training configuration
        filename: Checkpoint filename
        upload_to_hf: Whether to upload to Hugging Face after saving (async)
        upload_threads: Optional list to track upload threads (for cleanup)
        scheduler: Optional scheduler whose state should be checkpointed
        best_val_loss: Best validation loss observed so far
        training_start_time: ISO format timestamp when training started
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config.__dict__,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
    if training_start_time is not None:
        checkpoint['training_start_time'] = training_start_time

    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")

    # Upload to Hugging Face asynchronously if enabled
    if upload_to_hf and config.hf_auto_upload:
        def upload_in_background():
            try:
                print(f"[Background] Uploading {filename} to Hugging Face...")
                model_config = {
                    'hidden_size': getattr(config, 'hidden_size', 256),
                    'hidden2_size': getattr(config, 'hidden2_size', 32),
                    'hidden3_size': getattr(config, 'hidden3_size', 32),
                }
                upload_model_to_hf(
                    checkpoint_path=filepath,
                    config=config,
                    model_config=model_config,
                    training_start_time=training_start_time,
                )
                print(f"[Background] Upload of {filename} completed!")
            except Exception as e:
                print(f"[Background] Warning: Failed to upload {filename} to Hugging Face: {e}")

        # Start upload in background thread
        thread = threading.Thread(target=upload_in_background, daemon=False)
        thread.start()
        if upload_threads is not None:
            upload_threads.append(thread)
        print(f"Started background upload for {filename} (training continues...)")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    map_location: Optional[torch.device] = None,
    scheduler = None,
) -> Tuple[int, float, Optional[str]]:
    """
    Load model checkpoint

    Args:
        model: NNUE model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        map_location: torch.device or string for loading checkpoints
        scheduler: Optional scheduler to restore

    Returns:
        Tuple of (start_epoch, best_val_loss, training_start_time)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    stored_val_loss = checkpoint.get('val_loss')
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    training_start_time = checkpoint.get('training_start_time')

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    if stored_val_loss is not None:
        print(f"Validation loss: {stored_val_loss:.4f}")

    if best_val_loss != float('inf'):
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    return start_epoch, best_val_loss, training_start_time


def ensure_data_exists(config: TrainingConfig) -> bool:
    """
    Ensure training and validation data files exist, downloading if needed

    Args:
        config: Training configuration

    Returns:
        True if data exists or was successfully downloaded, False otherwise
    """
    train_exists = os.path.exists(config.train_data_path)
    val_exists = config.val_data_path and os.path.exists(config.val_data_path)

    if train_exists and (not config.val_data_path or val_exists):
        print(f"Data files found:")
        print(f"  Train: {config.train_data_path}")
        if config.val_data_path:
            print(f"  Val:   {config.val_data_path}")
        return True

    if not config.auto_download:
        print(f"Data files not found:")
        if not train_exists:
            print(f"  Missing: {config.train_data_path}")
        if config.val_data_path and not val_exists:
            print(f"  Missing: {config.val_data_path}")
        print("Set config.auto_download = True to automatically download data")
        return False

    print("Data files not found. Downloading and processing...")

    output_dir = os.path.dirname(config.train_data_path) or config.download_output_dir

    download_kwargs = {
        'output_dir': output_dir,
        'year': config.download_year,
        'month': config.download_month,
        'rated_only': config.download_rated_only,
        'stockfish_path': config.stockfish_path,
        'depth': config.download_depth,
        'max_games': config.download_max_games,
        'positions_per_game': config.download_positions_per_game,
        'max_positions': None,
        'num_workers': config.download_num_workers,
        'batch_size': config.download_batch_size,
        'download_mode': config.download_mode,
        'skip_redownload': config.download_skip_redownload,
    }

    try:
        output_path = download_and_process_lichess_data(**download_kwargs)

        if not output_path:
            print("Failed to download data")
            return False

        os.makedirs(output_dir, exist_ok=True)

        # Split dataset if needed
        if config.val_data_path:
            from train.dataset import split_dataset
            train_path, val_path = split_dataset(
                output_path,
                train_ratio=config.train_val_split_ratio,
                output_dir=output_dir,
                config=config
            )
            config.train_data_path = train_path
            config.val_data_path = val_path
        else:
            config.train_data_path = output_path

        print(f"Data ready:")
        print(f"  Train: {config.train_data_path}")
        if config.val_data_path:
            print(f"  Val:   {config.val_data_path}")
        return True

    except Exception as e:
        print(f"Error downloading data: {e}")
        return False


def train(config: TrainingConfig) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """
    Main training function

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, training_history)
    """
    if not ensure_data_exists(config):
        raise FileNotFoundError("Training data not found and could not be downloaded")

    set_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")
    print("Using NNUE-based model architecture")

    print("\nLoading data...")
    print(f"Batch size: {config.batch_size}")
    print(f"Num workers: {config.num_workers} (will be set to 0 for sparse encoding)")
    print(f"Device: {device}")
    train_loader, val_loader = create_dataloaders(
        train_path=config.train_data_path,
        val_path=config.val_data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_train_positions=config.max_train_positions,
        max_val_positions=config.max_val_positions,
        score_min=config.eval_score_min,
        score_max=config.eval_score_max,
        device=device,
        normalize=True  # Enable normalization to stabilize training
    )

    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")

    # Create NNUE model
    print("\nCreating NNUE model...")
    model = ChessNNUEModel(
        hidden_size=getattr(config, 'hidden_size', 256),
        hidden2_size=getattr(config, 'hidden2_size', 32),
        hidden3_size=getattr(config, 'hidden3_size', 32)
    )
    model = model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

    # Create optimizer
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Loss function
    criterion = get_loss_function(config)

    start_epoch = 0
    best_val_loss = float('inf')
    training_start_time = None

    if config.resume_from:
        start_epoch, best_val_loss, training_start_time = load_checkpoint(
            model,
            optimizer,
            config.resume_from,
            map_location=device,
            scheduler=scheduler,
        )
        # Sync scheduler with start_epoch
        if scheduler is not None:
            while (hasattr(scheduler, 'current_epoch') and
                   scheduler.current_epoch < start_epoch - 1):
                scheduler.step()

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    if config.warmup_epochs > 0:
        print(f"Warmup: {config.warmup_epochs} epochs (LR: {config.warmup_start_lr or 0.0} -> {config.learning_rate})")
    print("=" * 80)

    # Track training start time if not resuming from checkpoint
    from datetime import datetime
    if training_start_time is None:
        training_start_time = datetime.now().isoformat()

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    # Track upload threads for cleanup
    upload_threads: List[threading.Thread] = []
    has_validation = val_loader is not None

    last_val_loss: Optional[float] = None
    last_train_loss: Optional[float] = None

    # Early stopping tracking
    epochs_without_improvement = 0
    
    # Validation loss smoothing (exponential moving average) to reduce noise
    val_loss_smooth = None
    val_loss_smooth_alpha = 0.7  # Smoothing factor (0 = no smoothing, 1 = no change)
    best_val_loss_smooth = float('inf')  # Track best smoothed loss for early stopping

    # Initialize learning rate for first epoch if warmup is enabled and starting from beginning
    if scheduler is not None and config.warmup_epochs > 0 and start_epoch == 0:
        scheduler.step()  # Set LR for epoch 0

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch
        )
        training_history['train_loss'].append(train_loss)
        last_train_loss = train_loss

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rate'].append(current_lr)

        # Show warmup status
        warmup_status = ""
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            warmup_status = f" [Warmup {epoch+1}/{config.warmup_epochs}]"

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}{warmup_status}")

        # Validate
        val_loss = None
        improved = False
        if has_validation and (epoch + 1) % config.validate_every_n_epochs == 0:
            val_loss = validate(model, val_loader, criterion, device, config)
            training_history['val_loss'].append(val_loss)
            last_val_loss = val_loss
            
            # Apply exponential moving average smoothing
            if val_loss_smooth is None:
                val_loss_smooth = val_loss
            else:
                val_loss_smooth = val_loss_smooth_alpha * val_loss_smooth + (1 - val_loss_smooth_alpha) * val_loss
            
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f} (smoothed: {val_loss_smooth:.4f})")
            
            # Track best model based on raw validation loss (more accurate for checkpointing)
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Store raw loss for checkpointing
                improved = True
            
            # Use smoothed loss for early stopping decisions (more stable, less noise)
            if val_loss_smooth < best_val_loss_smooth:
                best_val_loss_smooth = val_loss_smooth
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping check based on smoothed loss
            if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered! No improvement for {config.early_stopping_patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

        # Save best checkpoint whenever validation improves
        if improved:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, 'best_model.pt', upload_to_hf=False,
                upload_threads=upload_threads, scheduler=scheduler,
                best_val_loss=best_val_loss, training_start_time=training_start_time,
            )

        # Save regular checkpoint
        save_regular = (epoch + 1) % config.save_every_n_epochs == 0
        if save_regular and (not config.save_best_only or not has_validation):
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, f'checkpoint_epoch_{epoch+1}.pt',
                upload_to_hf=config.hf_auto_upload, upload_threads=upload_threads,
                scheduler=scheduler, best_val_loss=best_val_loss,
                training_start_time=training_start_time,
            )

        # Update learning rate
        if scheduler:
            scheduler.step()

        print("-" * 80)

    if last_train_loss is None:
        print("\nNo new epochs were run (start_epoch >= num_epochs). Skipping additional checkpointing.")
        final_val_loss = last_val_loss
    else:
        # Run final validation if we have a val_loader and didn't just validate
        final_val_loss = last_val_loss
        final_improved = False
        if has_validation and (config.num_epochs % config.validate_every_n_epochs != 0):
            print("Running final validation...")
            final_val_loss = validate(model, val_loader, criterion, device, config)
            training_history['val_loss'].append(final_val_loss)
            last_val_loss = final_val_loss
            print(f"Final Validation Loss: {final_val_loss:.4f}")
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                final_improved = True

        # Update best model if final validation improved
        if final_improved:
            save_checkpoint(
                model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss,
                config, 'best_model.pt', upload_to_hf=False,
                upload_threads=upload_threads, scheduler=scheduler,
                best_val_loss=best_val_loss, training_start_time=training_start_time,
            )

        # Upload best model at the end (if enabled in config)
        if has_validation and best_val_loss < float('inf') and config.hf_auto_upload:
            best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                print("\nUploading best model to Hugging Face...")
                def upload_best_model():
                    try:
                        model_config = {
                            'hidden_size': getattr(config, 'hidden_size', 256),
                            'hidden2_size': getattr(config, 'hidden2_size', 32),
                            'hidden3_size': getattr(config, 'hidden3_size', 32),
                        }
                        upload_model_to_hf(
                            checkpoint_path=best_model_path,
                            config=config,
                            model_config=model_config,
                            training_start_time=training_start_time,
                        )
                        print("[Background] Best model upload completed!")
                    except Exception as e:
                        print(f"[Background] Warning: Failed to upload best model to Hugging Face: {e}")

                thread = threading.Thread(target=upload_best_model, daemon=False)
                thread.start()
                upload_threads.append(thread)
                print("Started background upload for best model (training continues...)")

        # Ensure last epoch checkpoint is saved
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss,
            config, f'checkpoint_epoch_{config.num_epochs}.pt',
            upload_to_hf=False, upload_threads=upload_threads,
            scheduler=scheduler, best_val_loss=best_val_loss,
            training_start_time=training_start_time,
        )

        # Save final model snapshot
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss,
            config, 'final_model.pt', upload_to_hf=False,
            upload_threads=upload_threads, scheduler=scheduler,
            best_val_loss=best_val_loss, training_start_time=training_start_time,
        )

    print("\nTraining complete!")
    if best_val_loss < float('inf'):
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print("Best validation loss: N/A (validation not run)")

    # Wait for any background uploads to complete
    if upload_threads:
        print(f"\nWaiting for {len(upload_threads)} background upload(s) to complete...")
        for thread in upload_threads:
            thread.join()
        print("All uploads completed!")

    return model, training_history


def main():
    """Main entry point"""
    config = get_config('default')
    config.__post_init__()

    print("NNUE Training Configuration")
    print("=" * 80)
    for field, value in config.__dict__.items():
        print(f"{field:30s}: {value}")
    print("=" * 80)

    train(config)


if __name__ == '__main__':
    main()

