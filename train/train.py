"""
Main training script for NNUE chess evaluation model
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

# Handle both direct execution and package import
try:
    from .model import NNUEModel, count_parameters
    from .dataset import create_dataloaders, split_dataset
    from .config import get_config, TrainingConfig
    from .download_data import download_and_process_lichess_data
    from .upload_to_hf import upload_model_to_hf
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train.model import NNUEModel, count_parameters
    from train.dataset import create_dataloaders, split_dataset
    from train.config import get_config, TrainingConfig
    from train.download_data import download_and_process_lichess_data
    from train.upload_to_hf import upload_model_to_hf


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
            momentum=0.9,  # Standard SGD momentum
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer: torch.optim.Optimizer, 
                  config: TrainingConfig) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create learning rate scheduler based on configuration"""
    if config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs
        )
    else:
        return None


def get_loss_function(config: TrainingConfig) -> torch.nn.Module:
    """Get loss function based on configuration"""
    if config.loss_function == 'mse':
        return nn.MSELoss()
    elif config.loss_function == 'huber':
        return nn.HuberLoss()
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

    for batch_idx, (white_features, black_features, targets) in enumerate(pbar):
        # Move to device
        white_features = white_features.to(device)
        black_features = black_features.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(white_features, black_features)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

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
        for white_features, black_features, targets in pbar:
            # Move to device
            white_features = white_features.to(device)
            black_features = black_features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(white_features, black_features)

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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    best_val_loss: Optional[float] = None,
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

    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")

    # Upload to Hugging Face asynchronously if enabled
    if upload_to_hf and config.hf_auto_upload:
        def upload_in_background():
            try:
                print(f"[Background] Uploading {filename} to Hugging Face...")
                model_config = {
                    'hidden_size': config.hidden_size,
                    'hidden2_size': config.hidden2_size,
                    'hidden3_size': config.hidden3_size,
                }
                upload_model_to_hf(
                    checkpoint_path=filepath,
                    config=config,
                    model_config=model_config,
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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[int, float]:
    """
    Load model checkpoint

    Args:
        model: NNUE model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        map_location: torch.device or string for loading checkpoints
        scheduler: Optional scheduler to restore

    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    stored_val_loss = checkpoint.get('val_loss')
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    if stored_val_loss is not None:
        print(f"Validation loss: {stored_val_loss:.4f}")

    if best_val_loss != float('inf'):
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


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
        'max_games_searched': config.download_max_games_searched,
        'positions_per_game': config.download_positions_per_game,
        'max_positions': None,
        'output_format': config.download_output_format,
        'num_workers': config.download_num_workers,
        'batch_size': config.download_batch_size,
        'download_mode': config.download_mode,
        'skip_filter': config.download_skip_filter,
        'skip_redownload': config.download_skip_redownload,
    }
    
    try:
        output_path = download_and_process_lichess_data(**download_kwargs)
        
        if not output_path:
            print("Failed to download data")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not train_exists:
            if config.val_data_path:
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
        elif config.val_data_path and not val_exists:
            train_path, val_path = split_dataset(
                config.train_data_path,
                train_ratio=config.train_val_split_ratio,
                output_dir=output_dir,
                config=config
            )
            config.train_data_path = train_path
            config.val_data_path = val_path
        
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

    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_path=config.train_data_path,
        val_path=config.val_data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_train_positions=config.max_train_positions,
        max_val_positions=config.max_val_positions
    )

    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = NNUEModel(
        hidden_size=config.hidden_size,
        hidden2_size=config.hidden2_size,
        hidden3_size=config.hidden3_size
    )
    model = model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Loss function
    criterion = get_loss_function(config)

    start_epoch = 0
    best_val_loss = float('inf')

    if config.resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            model,
            optimizer,
            config.resume_from,
            map_location=device,
            scheduler=scheduler,
        )

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("=" * 80)

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

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Validate
        val_loss = None
        improved = False
        if has_validation and (epoch + 1) % config.validate_every_n_epochs == 0:
            val_loss = validate(model, val_loader, criterion, device, config)
            training_history['val_loss'].append(val_loss)
            last_val_loss = val_loss
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True

        # Save best checkpoint whenever validation improves
        if improved:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, 'best_model.pt', upload_to_hf=True,
                upload_threads=upload_threads, scheduler=scheduler,
                best_val_loss=best_val_loss,
            )

        # Save regular checkpoint
        save_regular = (epoch + 1) % config.save_every_n_epochs == 0
        if save_regular and (not config.save_best_only or not has_validation):
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, f'checkpoint_epoch_{epoch+1}.pt',
                upload_to_hf=True, upload_threads=upload_threads,
                scheduler=scheduler, best_val_loss=best_val_loss,
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

        if final_improved:
            save_checkpoint(
                model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss,
                config, 'best_model.pt', upload_to_hf=True,
                upload_threads=upload_threads, scheduler=scheduler,
                best_val_loss=best_val_loss,
            )

        # Ensure last epoch checkpoint is saved
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss,
            config, f'checkpoint_epoch_{config.num_epochs}.pt',
            upload_to_hf=True, upload_threads=upload_threads,
            scheduler=scheduler, best_val_loss=best_val_loss,
        )

        # Save final model snapshot
        save_checkpoint(
            model, optimizer, config.num_epochs - 1, last_train_loss, final_val_loss,
            config, 'final_model.pt', upload_to_hf=True,
            upload_threads=upload_threads, scheduler=scheduler,
            best_val_loss=best_val_loss,
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
    for field, value in config.__dict__.items():
        print(f"{field:30s}: {value}")

    train(config)


if __name__ == '__main__':
    main()
