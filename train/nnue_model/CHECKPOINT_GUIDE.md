# Checkpoint Resumption Guide

## Overview

The training script now has robust checkpoint resumption capabilities. If training fails or is interrupted, you can easily resume from the last saved checkpoint.

## Features

✅ **Auto-detection** - Automatically finds latest checkpoint
✅ **Interactive prompt** - Asks if you want to resume
✅ **Command-line control** - Full CLI for automation
✅ **Error recovery** - Helpful messages if training fails
✅ **State preservation** - Resumes optimizer, scheduler, and training progress

## Quick Start

### 1. Start Training

```bash
python train/nnue_model/train.py
```

### 2. If Interrupted (Ctrl+C)

```
================================================================================
Training interrupted by user
================================================================================

You can resume training later with:
  python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_45.pt
  or
  python train/nnue_model/train.py --resume auto
```

### 3. Resume Training

```bash
# Auto-detect latest checkpoint
python train/nnue_model/train.py --resume auto

# Or specify exact checkpoint
python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_45.pt
```

## Usage Modes

### Interactive Mode (Default)

When you run the script and checkpoints exist, you'll be prompted:

```bash
python train/nnue_model/train.py
```

Output:
```
================================================================================
EXISTING CHECKPOINT FOUND
================================================================================
Checkpoint: checkpoint_epoch_45.pt
Last epoch: 44
Train loss: 0.2345
Val loss: 0.2567
================================================================================

Options:
  1. Resume training from this checkpoint
  2. Start fresh training (existing checkpoints will be kept)
  3. Exit
================================================================================

Your choice (1/2/3):
```

### Auto-Resume Mode

Automatically resume without prompt (useful for scripts):

```bash
python train/nnue_model/train.py --no-prompt
```

Or:

```bash
python train/nnue_model/train.py --resume auto
```

### Fresh Start Mode

Ignore existing checkpoints and start fresh:

```bash
python train/nnue_model/train.py --fresh
```

### Specific Checkpoint

Resume from a specific checkpoint file:

```bash
python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_30.pt
```

## Command-Line Arguments

```
usage: train.py [-h] [--config CONFIG] [--resume RESUME] [--no-prompt] [--fresh]

Train NNUE chess evaluation model

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Training configuration to use
                   choices: default, fast, quality, large_scale, rtx5070, rtx5070_quality
  --resume RESUME  Path to checkpoint to resume from (or "auto" to auto-detect)
  --no-prompt      Skip resume prompt and auto-resume if checkpoint exists
  --fresh          Start fresh training even if checkpoints exist
```

## Examples

### Example 1: Training Interrupted by Crash

```bash
# Start training
python train/nnue_model/train.py --config rtx5070

# ... training runs for 45 epochs then crashes ...

# Resume automatically
python train/nnue_model/train.py --resume auto
```

### Example 2: Switch Configuration Mid-Training

```bash
# Start with fast config
python train/nnue_model/train.py --config fast

# Later, resume with quality config
python train/nnue_model/train.py --config rtx5070_quality --resume auto
```

**Note**: Model architecture must match! You can change training hyperparameters (LR, batch size) but not model size.

### Example 3: Automated Training with Restarts

```bash
#!/bin/bash
# training_script.sh - Auto-restart on failure

while true; do
    python train/nnue_model/train.py --no-prompt

    # Check exit code
    if [ $? -eq 0 ]; then
        echo "Training completed successfully!"
        break
    else
        echo "Training failed, restarting in 5 seconds..."
        sleep 5
    fi
done
```

### Example 4: Continue After Early Stopping

```bash
# Training stopped early at epoch 60 due to no improvement

# Resume and train longer
python train/nnue_model/train.py --resume auto

# Or modify config first
python -c "
from train.config import get_config
from train.nnue_model.train import train

config = get_config('rtx5070')
config.resume_from = 'checkpoints/checkpoint_epoch_60.pt'
config.num_epochs = 150  # Train to epoch 150 instead of 100
config.early_stopping_patience = 30  # More patience
train(config)
"
```

## What Gets Saved in Checkpoints

Each checkpoint contains:

```python
{
    'epoch': 44,                          # Last completed epoch
    'model_state_dict': {...},            # Model weights
    'optimizer_state_dict': {...},        # Optimizer state (momentum, etc.)
    'scheduler_state_dict': {...},        # LR scheduler state
    'train_loss': 0.2345,                 # Last training loss
    'val_loss': 0.2567,                   # Last validation loss
    'best_val_loss': 0.2500,              # Best validation loss so far
    'config': {...},                      # Training configuration
    'training_start_time': '2024-01-15T10:30:00'  # When training began
}
```

## What Gets Restored

When resuming:

✅ **Model weights** - Exact model state
✅ **Optimizer state** - Momentum, adaptive learning rates
✅ **LR scheduler** - Warmup/decay schedule position
✅ **Training progress** - Starts from next epoch
✅ **Best model tracking** - Continues tracking best validation loss
✅ **Random state** - For reproducibility (seed applied)

## Checkpoint Files

Default checkpoint directory: `checkpoints/`

Files created:
- `checkpoint_epoch_5.pt` - Every 5 epochs (configurable)
- `checkpoint_epoch_10.pt`
- `best_model.pt` - Best validation loss
- `final_model.pt` - Last epoch

**Auto-resume uses**: Most recent `checkpoint_epoch_N.pt` file

## Troubleshooting

### "Checkpoint not found"

```bash
# List available checkpoints
ls -lh checkpoints/

# Use specific checkpoint
python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_XX.pt
```

### "Model architecture mismatch"

```
RuntimeError: Error loading state_dict...
```

**Cause**: Checkpoint has different model size than current config

**Fix**: Use same config as original training:
```bash
python train/nnue_model/train.py --config rtx5070 --resume auto
```

### "Out of memory after resuming"

**Cause**: Optimizer state requires more memory

**Fix**: Reduce batch size slightly:
```python
config.batch_size = 3072  # Was 4096
```

### Training starts from epoch 0 instead of resuming

**Cause**: `--fresh` flag or no checkpoint found

**Fix**: Check checkpoint exists and don't use `--fresh`:
```bash
ls checkpoints/checkpoint_epoch_*.pt
python train/nnue_model/train.py --resume auto
```

## Best Practices

### 1. Regular Checkpoints

Keep default `save_every_n_epochs=5` or make it more frequent:

```python
config.save_every_n_epochs = 2  # Save every 2 epochs
```

### 2. Keep Multiple Checkpoints

Don't use `save_best_only=True` during long training:

```python
config.save_best_only = False  # Keep all epoch checkpoints
```

### 3. Backup Checkpoints

For critical training runs:

```bash
# Periodically backup
cp -r checkpoints/ checkpoints_backup_$(date +%Y%m%d)/
```

### 4. Monitor Training

Use tmux/screen for long runs:

```bash
# Start in tmux
tmux new -s training
python train/nnue_model/train.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### 5. Use Auto-Resume for Clusters

On compute clusters with time limits:

```bash
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Will auto-resume if job restarts
python train/nnue_model/train.py --no-prompt
```

## Testing Checkpoint Resumption

Quick test to verify it works:

```python
# test_checkpoint.py
from train.config import get_config
from train.nnue_model.train import train

# Train for 5 epochs
config = get_config('fast')
config.num_epochs = 5
config.save_every_n_epochs = 2
train(config)

# Resume for 5 more
config.resume_from = 'checkpoints/checkpoint_epoch_4.pt'
config.num_epochs = 10  # Will train epochs 5-10
train(config)
```

## Advanced: Programmatic Resumption

```python
from train.nnue_model.train import train, find_latest_checkpoint
from train.config import get_config

config = get_config('rtx5070')

# Auto-find and resume
latest = find_latest_checkpoint(config.checkpoint_dir)
if latest:
    print(f"Resuming from: {latest}")
    config.resume_from = latest
else:
    print("Starting fresh training")

train(config)
```

## Summary

**To resume training after failure:**

1. **Automatic** (recommended):
   ```bash
   python train/nnue_model/train.py --resume auto
   ```

2. **Interactive**:
   ```bash
   python train/nnue_model/train.py
   # Choose option 1 when prompted
   ```

3. **Specific checkpoint**:
   ```bash
   python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_X.pt
   ```

Training will continue from the saved epoch with all state preserved! 🎯
