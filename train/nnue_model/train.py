"""
Main training script for NNUE-based chess evaluation model
Uses bitmap/bitboard features for efficient representation.
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
    from .model import ChessNNUEModel, BitboardFeatures, count_parameters
    from ..config import get_config, TrainingConfig
    from ..download_data import download_and_process_lichess_data
    from ..upload_to_hf import upload_model_to_hf
    from ..dataset import ChessPositionDatasetSparse, collate_sparse, create_dataloaders_sparse
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from train.nnue_model.model import ChessNNUEModel, BitboardFeatures, count_parameters
    from train.config import get_config, TrainingConfig
    from train.download_data import download_and_process_lichess_data
    from train.upload_to_hf import upload_model_to_hf
    from train.dataset import ChessPositionDatasetSparse, collate_sparse, create_dataloaders_sparse


class ChessBoardDatasetBitmap(Dataset):
    """
    PyTorch Dataset for chess positions with evaluations using bitmap NNUE features
    Includes normalization support

    Expected data format (JSONL - one JSON object per line):
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "eval": 0.15}
        {"fen": "...", "eval": -0.5}
        ...
    """

    def __init__(self, data_path: str, max_positions: Optional[int] = None,
                 score_min: Optional[float] = None, score_max: Optional[float] = None,
                 normalize: bool = True):
        """Initialize bitmap dataset"""
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
            print(f"Loaded {len(self.positions)} positions from {data_path} (bitmap mode)")
            print(f"Evaluation stats: mean={self.eval_mean:.2f}, std={self.eval_std:.2f}")
        else:
            print(f"Loaded {len(self.positions)} positions from {data_path} (bitmap mode)")

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Get a training sample with bitmap encoding

        Returns:
            Tuple of (bitmap_features, evaluation)
            where bitmap_features is a tensor of shape [768]
            (12 bitboards × 64 squares ordered by side to move)
        """
        fen = self.positions[idx]
        eval_score = self.evaluations[idx]

        # Parse FEN to board
        board = chess.Board(fen)

        # Get bitmap features
        features = BitboardFeatures.board_to_features_for_side(board, perspective=board.turn)

        # If it's black to move, flip the evaluation
        if board.turn == chess.BLACK:
            eval_score = -eval_score

        # Normalize evaluation if enabled
        if self.normalize:
            eval_score = (eval_score - self.eval_mean) / self.eval_std

        return features, eval_score


def collate_fn_bitmap(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for bitmap batching with normalized evaluations

    Returns:
        Tuple of (features, evaluations)
        where features is [batch_size, 768] and evaluations is [batch_size, 1]
    """
    features_list = []
    evaluations = []

    for features, eval_score in batch:
        features_list.append(features)
        evaluations.append(eval_score)

    # Stack features and evaluations
    features = torch.stack(features_list, dim=0)  # [batch_size, 768]
    evaluations = torch.tensor(evaluations, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1]

    return features, evaluations


def create_dataloaders(train_path: str, val_path: Optional[str] = None,
                       batch_size: int = 256, num_workers: int = 0,
                       max_train_positions: Optional[int] = None,
                       max_val_positions: Optional[int] = None,
                       score_min: Optional[float] = None,
                       score_max: Optional[float] = None,
                       device: Optional[torch.device] = None,
                       normalize: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders with bitmap encoding

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_path not provided
    """
    # Create training dataset
    train_dataset = ChessBoardDatasetBitmap(train_path, max_positions=max_train_positions,
                                           score_min=score_min, score_max=score_max,
                                           normalize=normalize)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_bitmap,
        pin_memory=True if device and device.type == 'cuda' else False
    )

    # Create validation dataset if path provided
    val_loader = None
    if val_path:
        # Use same normalization stats from training set
        val_dataset = ChessBoardDatasetBitmap(val_path, max_positions=max_val_positions,
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
            collate_fn=collate_fn_bitmap,
            pin_memory=True if device and device.type == 'cuda' else False
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
                device: torch.device, config: TrainingConfig, epoch: int,
                eval_std: Optional[float] = None) -> Dict[str, float]:
    """
    Train for one epoch with comprehensive metrics

    Args:
        model: NNUE model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        config: Training configuration
        epoch: Current epoch number
        eval_std: Standard deviation of evaluations in centipawns (for denormalization)

    Returns:
        Dictionary with training metrics:
        - loss: Average loss (normalized)
        - rmse_normalized: RMSE in normalized space
        - rmse_centipawns: RMSE in centipawns (if eval_std provided)
        - mae_normalized: MAE in normalized space
        - mae_centipawns: MAE in centipawns (if eval_std provided)
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    # Track predictions and targets for metrics (sample every N batches to save memory)
    sample_outputs = []
    sample_targets = []
    sample_interval = max(1, num_batches // 10)  # Sample ~10 batches per epoch

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}',
                disable=not config.verbose)

    for batch_idx, (features, targets) in enumerate(pbar):
        # Move to device
        features = features.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)
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
        
        # Sample predictions for metrics (to avoid memory issues)
        if batch_idx % sample_interval == 0:
            sample_outputs.append(outputs.detach().cpu())
            sample_targets.append(targets.detach().cpu())

        # Update progress bar
        if batch_idx % config.log_every_n_batches == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics from samples
    if sample_outputs:
        sample_outputs = torch.cat(sample_outputs, dim=0).squeeze()
        sample_targets = torch.cat(sample_targets, dim=0).squeeze()
        errors = sample_outputs - sample_targets
        errors_abs = torch.abs(errors)
        
        rmse_normalized = torch.sqrt(torch.mean(errors ** 2)).item()
        mae_normalized = torch.mean(errors_abs).item()
    else:
        rmse_normalized = 0.0
        mae_normalized = 0.0
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        'loss': avg_loss,
        'rmse_normalized': rmse_normalized,
        'mae_normalized': mae_normalized,
    }
    
    # Convert to centipawns if eval_std is available
    if eval_std is not None and eval_std > 0:
        metrics['rmse_centipawns'] = rmse_normalized * eval_std
        metrics['mae_centipawns'] = mae_normalized * eval_std
    
    return metrics


def validate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module, device: torch.device,
             config: TrainingConfig, eval_std: Optional[float] = None) -> Dict[str, float]:
    """
    Validate the model with comprehensive metrics

    Args:
        model: NNUE model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        config: Training configuration
        eval_std: Standard deviation of evaluations in centipawns (for denormalization)

    Returns:
        Dictionary with validation metrics:
        - loss: Average loss (normalized)
        - rmse_normalized: RMSE in normalized space
        - rmse_centipawns: RMSE in centipawns (if eval_std provided)
        - mae_normalized: MAE in normalized space
        - mae_centipawns: MAE in centipawns (if eval_std provided)
        - accuracy_50cp: % of predictions within 50 centipawns
        - accuracy_100cp: % of predictions within 100 centipawns
        - accuracy_200cp: % of predictions within 200 centipawns
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    # Track predictions and targets for detailed metrics
    all_outputs = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', disable=not config.verbose)
        for features, targets in pbar:
            # Move to device
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(features)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Collect predictions and targets for metrics
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            total_samples += targets.size(0)

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    # Concatenate all predictions and targets
    all_outputs = torch.cat(all_outputs, dim=0).squeeze()
    all_targets = torch.cat(all_targets, dim=0).squeeze()
    
    # Calculate metrics
    errors = all_outputs - all_targets
    errors_abs = torch.abs(errors)
    
    # Loss (already averaged)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # RMSE (Root Mean Squared Error)
    rmse_normalized = torch.sqrt(torch.mean(errors ** 2)).item()
    
    # MAE (Mean Absolute Error)
    mae_normalized = torch.mean(errors_abs).item()
    
    metrics = {
        'loss': avg_loss,
        'rmse_normalized': rmse_normalized,
        'mae_normalized': mae_normalized,
    }
    
    # Convert to centipawns if eval_std is available
    if eval_std is not None and eval_std > 0:
        metrics['rmse_centipawns'] = rmse_normalized * eval_std
        metrics['mae_centipawns'] = mae_normalized * eval_std
        
        # Accuracy metrics (percentage within X centipawns)
        errors_centipawns = errors_abs * eval_std
        metrics['accuracy_50cp'] = (errors_centipawns <= 50).float().mean().item() * 100
        metrics['accuracy_100cp'] = (errors_centipawns <= 100).float().mean().item() * 100
        metrics['accuracy_200cp'] = (errors_centipawns <= 200).float().mean().item() * 100
        
        # Additional statistics
        metrics['max_error_centipawns'] = errors_centipawns.max().item()
        metrics['median_error_centipawns'] = errors_centipawns.median().item()
        metrics['p95_error_centipawns'] = torch.quantile(errors_centipawns, 0.95).item()
    
    return metrics


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
    print(f"Num workers: {config.num_workers}")
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
    
    # Extract eval_std from dataset for metric conversion
    eval_std = None
    if hasattr(train_loader.dataset, 'eval_std') and train_loader.dataset.eval_std > 0:
        eval_std = train_loader.dataset.eval_std
        eval_mean = train_loader.dataset.eval_mean
        print(f"\nEvaluation statistics:")
        print(f"  Mean: {eval_mean:.2f} centipawns")
        print(f"  Std:  {eval_std:.2f} centipawns")
        print(f"  (Metrics will be converted to centipawns)")

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
        'train_rmse_normalized': [],
        'train_mae_normalized': [],
        'train_rmse_centipawns': [],
        'train_mae_centipawns': [],
        'val_loss': [],
        'val_rmse_normalized': [],
        'val_mae_normalized': [],
        'val_rmse_centipawns': [],
        'val_mae_centipawns': [],
        'val_accuracy_50cp': [],
        'val_accuracy_100cp': [],
        'val_accuracy_200cp': [],
        'val_median_error_centipawns': [],
        'val_p95_error_centipawns': [],
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
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch, eval_std
        )
        training_history['train_loss'].append(train_metrics['loss'])
        training_history['train_rmse_normalized'].append(train_metrics['rmse_normalized'])
        training_history['train_mae_normalized'].append(train_metrics['mae_normalized'])
        if 'rmse_centipawns' in train_metrics:
            training_history['train_rmse_centipawns'].append(train_metrics['rmse_centipawns'])
            training_history['train_mae_centipawns'].append(train_metrics['mae_centipawns'])
        last_train_loss = train_metrics['loss']

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rate'].append(current_lr)

        # Show warmup status
        warmup_status = ""
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            warmup_status = f" [Warmup {epoch+1}/{config.warmup_epochs}]"

        # Print training metrics
        train_loss_str = f"Loss: {train_metrics['loss']:.4f}"
        if 'rmse_centipawns' in train_metrics:
            train_loss_str += f" | RMSE: {train_metrics['rmse_centipawns']:.1f}cp | MAE: {train_metrics['mae_centipawns']:.1f}cp"
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train {train_loss_str}, LR: {current_lr:.6f}{warmup_status}")

        # Validate
        val_metrics = None
        improved = False
        if has_validation and (epoch + 1) % config.validate_every_n_epochs == 0:
            val_metrics = validate(model, val_loader, criterion, device, config, eval_std)
            val_loss = val_metrics['loss']
            training_history['val_loss'].append(val_loss)
            training_history['val_rmse_normalized'].append(val_metrics['rmse_normalized'])
            training_history['val_mae_normalized'].append(val_metrics['mae_normalized'])
            if 'rmse_centipawns' in val_metrics:
                training_history['val_rmse_centipawns'].append(val_metrics['rmse_centipawns'])
                training_history['val_mae_centipawns'].append(val_metrics['mae_centipawns'])
                training_history['val_accuracy_50cp'].append(val_metrics.get('accuracy_50cp', 0))
                training_history['val_accuracy_100cp'].append(val_metrics.get('accuracy_100cp', 0))
                training_history['val_accuracy_200cp'].append(val_metrics.get('accuracy_200cp', 0))
                training_history['val_median_error_centipawns'].append(val_metrics.get('median_error_centipawns', 0))
                training_history['val_p95_error_centipawns'].append(val_metrics.get('p95_error_centipawns', 0))
            last_val_loss = val_loss
            
            # Apply exponential moving average smoothing
            if val_loss_smooth is None:
                val_loss_smooth = val_loss
            else:
                val_loss_smooth = val_loss_smooth_alpha * val_loss_smooth + (1 - val_loss_smooth_alpha) * val_loss
            
            # Print validation metrics
            val_loss_str = f"Loss: {val_loss:.4f} (smoothed: {val_loss_smooth:.4f})"
            if 'rmse_centipawns' in val_metrics:
                val_loss_str += f"\n  RMSE: {val_metrics['rmse_centipawns']:.1f}cp | MAE: {val_metrics['mae_centipawns']:.1f}cp"
                val_loss_str += f" | Median: {val_metrics.get('median_error_centipawns', 0):.1f}cp | P95: {val_metrics.get('p95_error_centipawns', 0):.1f}cp"
                val_loss_str += f"\n  Accuracy: {val_metrics.get('accuracy_50cp', 0):.1f}% within 50cp | {val_metrics.get('accuracy_100cp', 0):.1f}% within 100cp | {val_metrics.get('accuracy_200cp', 0):.1f}% within 200cp"
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val {val_loss_str}")
            
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
                model, optimizer, epoch, train_metrics['loss'], val_metrics['loss'] if val_metrics else None,
                config, 'best_model.pt', upload_to_hf=False,
                upload_threads=upload_threads, scheduler=scheduler,
                best_val_loss=best_val_loss, training_start_time=training_start_time,
            )

        # Save regular checkpoint
        save_regular = (epoch + 1) % config.save_every_n_epochs == 0
        if save_regular and (not config.save_best_only or not has_validation):
            save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'], val_metrics['loss'] if val_metrics else None,
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
        final_val_metrics = None
    else:
        # Run final validation if we have a val_loader and didn't just validate
        final_val_metrics = None
        final_improved = False
        if has_validation and (config.num_epochs % config.validate_every_n_epochs != 0):
            print("Running final validation...")
            final_val_metrics = validate(model, val_loader, criterion, device, config, eval_std)
            final_val_loss = final_val_metrics['loss']
            training_history['val_loss'].append(final_val_loss)
            training_history['val_rmse_normalized'].append(final_val_metrics['rmse_normalized'])
            training_history['val_mae_normalized'].append(final_val_metrics['mae_normalized'])
            if 'rmse_centipawns' in final_val_metrics:
                training_history['val_rmse_centipawns'].append(final_val_metrics['rmse_centipawns'])
                training_history['val_mae_centipawns'].append(final_val_metrics['mae_centipawns'])
                training_history['val_accuracy_50cp'].append(final_val_metrics.get('accuracy_50cp', 0))
                training_history['val_accuracy_100cp'].append(final_val_metrics.get('accuracy_100cp', 0))
                training_history['val_accuracy_200cp'].append(final_val_metrics.get('accuracy_200cp', 0))
                training_history['val_median_error_centipawns'].append(final_val_metrics.get('median_error_centipawns', 0))
                training_history['val_p95_error_centipawns'].append(final_val_metrics.get('p95_error_centipawns', 0))
            last_val_loss = final_val_loss
            
            # Print final validation metrics
            final_val_str = f"Loss: {final_val_loss:.4f}"
            if 'rmse_centipawns' in final_val_metrics:
                final_val_str += f" | RMSE: {final_val_metrics['rmse_centipawns']:.1f}cp | MAE: {final_val_metrics['mae_centipawns']:.1f}cp"
                final_val_str += f" | Median: {final_val_metrics.get('median_error_centipawns', 0):.1f}cp | P95: {final_val_metrics.get('p95_error_centipawns', 0):.1f}cp"
                final_val_str += f"\n  Accuracy: {final_val_metrics.get('accuracy_50cp', 0):.1f}% within 50cp | {final_val_metrics.get('accuracy_100cp', 0):.1f}% within 100cp | {final_val_metrics.get('accuracy_200cp', 0):.1f}% within 200cp"
            print(f"Final Validation - {final_val_str}")
            
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                final_improved = True

        # Update best model if final validation improved
        if final_improved:
            save_checkpoint(
                model, optimizer, config.num_epochs - 1, last_train_loss, final_val_metrics['loss'] if final_val_metrics else None,
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
        final_val_loss_for_checkpoint = final_val_metrics['loss'] if final_val_metrics else last_val_loss
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss_for_checkpoint,
            config, f'checkpoint_epoch_{config.num_epochs}.pt',
            upload_to_hf=False, upload_threads=upload_threads,
            scheduler=scheduler, best_val_loss=best_val_loss,
            training_start_time=training_start_time,
        )

        # Save final model snapshot
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss_for_checkpoint,
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


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the checkpoint directory

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Look for checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoint_files.append(filepath)

    if not checkpoint_files:
        return None

    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return checkpoint_files[0]


def prompt_resume_training(checkpoint_dir: str) -> Optional[str]:
    """
    Check for existing checkpoints and prompt user to resume

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to checkpoint to resume from, or None to start fresh
    """
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is None:
        return None

    # Load checkpoint to get info
    try:
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        epoch = checkpoint.get('epoch', 'unknown')
        train_loss = checkpoint.get('train_loss', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')

        print("\n" + "=" * 80)
        print("EXISTING CHECKPOINT FOUND")
        print("=" * 80)
        print(f"Checkpoint: {os.path.basename(latest_checkpoint)}")
        print(f"Last epoch: {epoch}")
        print(f"Train loss: {train_loss}")
        print(f"Val loss: {val_loss}")
        print("=" * 80)
        print("\nOptions:")
        print("  1. Resume training from this checkpoint")
        print("  2. Start fresh training (existing checkpoints will be kept)")
        print("  3. Exit")
        print("=" * 80)

        # Auto-resume if running in non-interactive mode
        if not sys.stdin.isatty():
            print("Non-interactive mode detected - auto-resuming from checkpoint")
            return latest_checkpoint

        choice = input("\nYour choice (1/2/3): ").strip()

        if choice == '1':
            print(f"\n✓ Resuming from: {latest_checkpoint}")
            return latest_checkpoint
        elif choice == '2':
            print("\n✓ Starting fresh training")
            return None
        else:
            print("\n✓ Exiting")
            sys.exit(0)

    except Exception as e:
        print(f"Warning: Could not load checkpoint {latest_checkpoint}: {e}")
        return None


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train NNUE chess evaluation model')
    parser.add_argument('--config', type=str, default='rtx5070',
                       choices=['default', 'fast', 'quality', 'large_scale', 'rtx5070', 'rtx5070_quality'],
                       help='Training configuration to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (or "auto" to auto-detect)')
    parser.add_argument('--no-prompt', action='store_true',
                       help='Skip resume prompt and auto-resume if checkpoint exists')
    parser.add_argument('--fresh', action='store_true',
                       help='Start fresh training even if checkpoints exist')

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)
    config.__post_init__()

    # Handle checkpoint resumption
    if args.fresh:
        print("\n✓ Starting fresh training (--fresh flag set)")
        config.resume_from = None
    elif args.resume:
        if args.resume == 'auto':
            latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
            if latest_checkpoint:
                print(f"\n✓ Auto-resuming from: {latest_checkpoint}")
                config.resume_from = latest_checkpoint
            else:
                print("\n✓ No checkpoint found - starting fresh training")
                config.resume_from = None
        else:
            if os.path.exists(args.resume):
                print(f"\n✓ Resuming from: {args.resume}")
                config.resume_from = args.resume
            else:
                print(f"\n✗ Checkpoint not found: {args.resume}")
                sys.exit(1)
    elif args.no_prompt:
        # Auto-resume without prompt
        latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
        if latest_checkpoint:
            print(f"\n✓ Auto-resuming from: {latest_checkpoint}")
            config.resume_from = latest_checkpoint
    else:
        # Interactive prompt
        resume_checkpoint = prompt_resume_training(config.checkpoint_dir)
        if resume_checkpoint:
            config.resume_from = resume_checkpoint

    print("\nNNUE Training Configuration")
    print("=" * 80)
    for field, value in config.__dict__.items():
        print(f"{field:30s}: {value}")
    print("=" * 80)

    try:
        train(config)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            print(f"\nYou can resume training later with:")
            print(f"  python train/nnue_model/train.py --resume {latest}")
            print(f"  or")
            print(f"  python train/nnue_model/train.py --resume auto")
        sys.exit(0)
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("Training failed with error:")
        print("=" * 80)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            print(f"\nYou can try resuming from the last checkpoint:")
            print(f"  python train/nnue_model/train.py --resume {latest}")
            print(f"  or")
            print(f"  python train/nnue_model/train.py --resume auto")
        sys.exit(1)


if __name__ == '__main__':
    main()
