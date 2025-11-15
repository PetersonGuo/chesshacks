"""
Main training script for NNUE chess evaluation model
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from model import NNUEModel, count_parameters
from dataset import create_dataloaders
from config import get_config, TrainingConfig


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, config: TrainingConfig):
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


def get_scheduler(optimizer, config: TrainingConfig):
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


def get_loss_function(config: TrainingConfig):
    """Get loss function based on configuration"""
    if config.loss_function == 'mse':
        return nn.MSELoss()
    elif config.loss_function == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}")


def train_epoch(model, train_loader, optimizer, criterion, device, config, epoch):
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

    return total_loss / num_batches


def validate(model, val_loader, criterion, device, config):
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

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filename):
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
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config.__dict__,
    }

    filepath = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    """
    Load model checkpoint

    Args:
        model: NNUE model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        map_location: torch.device or string for loading checkpoints

    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    stored_val_loss = checkpoint.get('val_loss')
    best_val_loss = float('inf') if stored_val_loss is None else stored_val_loss

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    if stored_val_loss is not None:
        print(f"Validation loss: {stored_val_loss:.4f}")

    return start_epoch, best_val_loss


def train(config: TrainingConfig, resume_from=None):
    """
    Main training function

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Set random seed
    set_seed(config.seed)

    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load data
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

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            model,
            optimizer,
            resume_from,
            map_location=device
        )

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("=" * 80)

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  device, config, epoch)
        training_history['train_loss'].append(train_loss)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rate'].append(current_lr)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Validate
        val_loss = None
        if val_loader and (epoch + 1) % config.validate_every_n_epochs == 0:
            val_loss = validate(model, val_loader, criterion, device, config)
            training_history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f}")

            # Save best model
            if config.save_best_only and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                              config, 'best_model.pt')

        # Save regular checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                          config, f'checkpoint_epoch_{epoch+1}.pt')

        # Update learning rate
        if scheduler:
            scheduler.step()

        print("-" * 80)

    # Save final model
    save_checkpoint(model, optimizer, config.num_epochs - 1, train_loss, val_loss,
                   config, 'final_model.pt')

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, training_history


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train NNUE chess evaluation model')

    # Configuration
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'fast', 'quality', 'large_scale'],
                       help='Configuration preset to use')

    # Data paths
    parser.add_argument('--train-data', type=str, default=None,
                       help='Path to training data file')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data file')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')

    # Device
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to train on')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load base configuration
    config = get_config(args.config)

    # Override with command line arguments
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.num_epochs = args.epochs
    if args.device:
        config.device = args.device
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    # Re-run validation and directory setup after overrides
    config.__post_init__()

    # Print configuration
    print("=" * 80)
    print("NNUE Training Configuration")
    print("=" * 80)
    for field, value in config.__dict__.items():
        print(f"{field:30s}: {value}")
    print("=" * 80)

    # Train
    train(config, resume_from=args.resume_from)


if __name__ == '__main__':
    main()
