# Ready to Train - Pre-flight Checklist

## ✓ System Status: READY

### What's Ready:

✅ **Bitmap NNUE Model** - Fully implemented and tested
✅ **RTX 5070 Configuration** - Optimized for GPU training
✅ **Training Pipeline** - All components verified
✅ **FEN → Bitboard Conversion** - Automatic during training
✅ **Syntax Check** - All Python files compile successfully

### What You Need:

❌ **Training Data** - No data found in `data/` directory

## Quick Start Options

### Option 1: Auto-Download Data (Recommended)

The config has `auto_download=True` enabled, so just run:

```bash
python train/nnue_model/train.py
```

This will:
1. Auto-download Lichess games (default: 100,000 games)
2. Extract positions and evaluate with Stockfish
3. Save to `data/lichess_evaluated_YYYY-MM.jsonl`
4. Start training automatically

**Note**: This will take 30-60 minutes for data preparation on first run.

### Option 2: Use Existing Data

If you already have data:

```bash
# Put your JSONL file in data/
# Format: {"fen": "...", "eval": 0.15}
cp your_data.jsonl data/train.jsonl

# Run training
python train/nnue_model/train.py
```

### Option 3: Quick Test with Small Dataset

For testing, use the fast config:

```python
from train.nnue_model.train import train
from train.config import get_config

config = get_config('fast')
config.download_max_games = 1000  # Small dataset
config.num_epochs = 5
config.auto_download = True

train(config)
```

## Current Configuration

The default config in `train.py` is set to **RTX 5070**:

```python
config = get_config('rtx5070')
```

### RTX 5070 Settings:
- **Device**: CUDA (auto-falls back to MPS/CPU if unavailable)
- **Batch Size**: 4096
- **Learning Rate**: 0.0006
- **Epochs**: 100
- **Model**: 512 → 64 → 64 (+ residual) — 440K params
- **Auto Download**: ✓ Enabled

## Expected Output

When you run training, you'll see:

```
NNUE Training Configuration
================================================================================
device                        : cuda
batch_size                    : 4096
learning_rate                 : 0.001
...
================================================================================

Checking for training data...
Data files not found. Downloading and processing...

Downloading Lichess database for 2024-11...
[Progress bar]

Extracting positions...
[Progress bar]

Evaluating with Stockfish...
[Progress bar]

Data ready:
  Train: data/train.jsonl
  Val:   data/val.jsonl

Loading data...
Loaded 90000 positions from data/train.jsonl (bitmap mode)
Loaded 10000 positions from data/val.jsonl (bitmap mode)

Creating NNUE model...
Model parameters: 439,617

Starting training for 100 epochs...
================================================================================
Epoch 1/100 - Train Loss: 1.2345, LR: 0.000020 [Warmup 1/5]
Epoch 1/100 - Val Loss: 1.3456
...
```

## Monitoring Training

### Check GPU Utilization
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Monitor Progress
Training progress is displayed with:
- Current epoch and loss
- Learning rate (with warmup indicator)
- Validation loss (every epoch)
- Early stopping countdown

### Checkpoints
Saved to `checkpoints/`:
- `best_model.pt` - Best validation loss
- `checkpoint_epoch_N.pt` - Every 5 epochs
- `final_model.pt` - Last epoch

## Checkpoint Resumption

### If Training is Interrupted

Training automatically saves checkpoints every 5 epochs. If interrupted:

```bash
# Auto-resume from latest checkpoint
python train/nnue_model/train.py --resume auto

# Or let it prompt you
python train/nnue_model/train.py
# Choose option 1 to resume

# Or specify exact checkpoint
python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_45.pt
```

### Command-Line Options

```bash
# Start fresh (ignore existing checkpoints)
python train/nnue_model/train.py --fresh

# Auto-resume without prompt (for scripts)
python train/nnue_model/train.py --no-prompt

# Use different config
python train/nnue_model/train.py --config rtx5070_quality --resume auto
```

**See**: `CHECKPOINT_GUIDE.md` for complete resumption documentation

## Troubleshooting

### "CUDA not available"
✓ Normal - will auto-fallback to MPS (Mac) or CPU
✓ Training will still work, just slower

### "Out of memory"
```python
config.batch_size = 2048  # Reduce from 4096
```

### "Training crashed/interrupted"
```bash
# Resume from last checkpoint
python train/nnue_model/train.py --resume auto
```

### "No Stockfish found"
```bash
# macOS
brew install stockfish

# Linux
sudo apt-get install stockfish

# Or specify path
config.stockfish_path = '/path/to/stockfish'
```

### "Download taking too long"
```python
config.download_max_games = 10000  # Reduce from 100K
config.download_positions_per_game = 10  # Reduce from 20
```

## Performance Expectations

### RTX 5070 (estimated):
- **Training speed**: ~50,000 positions/sec
- **Time per epoch**: ~2 seconds (100K positions)
- **Total time**: ~200 seconds for 100 epochs
- **With data download**: +30-60 minutes first time

### CPU/MPS:
- **Training speed**: ~5,000-10,000 positions/sec
- **Time per epoch**: ~10-20 seconds
- **Total time**: ~30 minutes for 100 epochs

## Ready to Go!

Just run:

```bash
python train/nnue_model/train.py
```

And grab a coffee while it downloads data! ☕

---

**Need help?** Check:
- `train/nnue_model/README_RTX5070.md` - Full documentation
- `train/config.py` - All available configs
- `train/nnue_model/test_training_pipeline.py` - Run tests
