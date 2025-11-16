# Quick Reference - NNUE Training

## Start Training

```bash
# Default (RTX 5070 config)
python train/nnue_model/train.py

# With specific config
python train/nnue_model/train.py --config rtx5070_quality

# Fast test
python train/nnue_model/train.py --config fast
```

## Resume Training

```bash
# Auto-detect and resume (recommended)
python train/nnue_model/train.py --resume auto

# Interactive prompt
python train/nnue_model/train.py
# Choose option 1

# Specific checkpoint
python train/nnue_model/train.py --resume checkpoints/checkpoint_epoch_45.pt

# Auto-resume for scripts
python train/nnue_model/train.py --no-prompt
```

## Fresh Start

```bash
# Ignore existing checkpoints
python train/nnue_model/train.py --fresh
```

## Available Configs

| Config | Batch Size | Model Size | Epochs | Best For |
|--------|-----------|------------|--------|----------|
| `fast` | 512 | 512→64→64 | 10 | Quick testing |
| `rtx5070` | 4096 | 512→64→64 | 100 | Fast training |
| `rtx5070_quality` | 2048 | 768→128→128 | 150 | Best accuracy |
| `quality` | 256 | 768→128→128 | 200 | Production |
| `default` | 512 | 512→64→64 | 30 | General use |

## Common Tasks

### Check Checkpoints
```bash
ls -lh checkpoints/
```

### Monitor Training
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training logs (if using tmux)
tmux attach -t training
```

### Evaluate Model
```python
import torch
import chess
from train.nnue_model.model import ChessNNUEModel

model = ChessNNUEModel(hidden_size=512, hidden2_size=64, hidden3_size=64)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

board = chess.Board()
score = model.evaluate_board(board)
print(f"Score: {score:.2f}")
```

### Change Config Mid-Training
```bash
# Resume with different hyperparameters
python train/nnue_model/train.py --config rtx5070_quality --resume auto
```

**Note**: Model architecture (hidden sizes) must match!

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Interrupt training (saves state) |
| `1` | Resume from checkpoint (when prompted) |
| `2` | Start fresh training (when prompted) |
| `3` | Exit (when prompted) |

## File Locations

```
train/nnue_model/
├── train.py              # Main training script
├── model.py              # Bitmap NNUE model
├── READY_TO_TRAIN.md     # Getting started guide
├── CHECKPOINT_GUIDE.md   # Resumption details
├── README_RTX5070.md     # RTX 5070 optimization
└── QUICK_REFERENCE.md    # This file

checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── best_model.pt         # Best validation loss
└── final_model.pt        # Last epoch

data/
├── train.jsonl           # Training data
└── val.jsonl             # Validation data
```

## Help Commands

```bash
# Show all options
python train/nnue_model/train.py --help

# Check Python syntax
python -m py_compile train/nnue_model/train.py

# Run tests
python train/nnue_model/test_training_pipeline.py

# Benchmark performance
python train/nnue_model/benchmark_bitmap.py
```

## Error Recovery

| Error | Solution |
|-------|----------|
| Training crashed | `python train/nnue_model/train.py --resume auto` |
| Out of memory | Use smaller batch size or model |
| No data found | Will auto-download on first run |
| CUDA unavailable | Auto-falls back to CPU/MPS |

## Quick Tips

✅ **Use tmux/screen** for long training runs
✅ **Save checkpoints frequently** (`save_every_n_epochs=2`)
✅ **Monitor GPU** with `nvidia-smi`
✅ **Start with fast config** for testing
✅ **Use auto-resume** for clusters with time limits

## One-Liners

```bash
# Full training pipeline (auto-downloads data)
python train/nnue_model/train.py

# Fast 10-epoch test
python train/nnue_model/train.py --config fast --fresh

# Resume after crash
python train/nnue_model/train.py --resume auto

# Production quality training
python train/nnue_model/train.py --config rtx5070_quality

# Auto-restart on failure
while true; do python train/nnue_model/train.py --no-prompt || sleep 5; done
```

## Model Stats

**Bitmap NNUE (512→64→64 + residual)**:
- Parameters: 439,617
- Size: 1.68 MB
- Input: 768 features (12 bitboards, re-ordered by side-to-move)
- Dual feature towers with residual refinement
- Speed: ~820K positions/sec (batched)

**Quality NNUE (768→128→128 + residual)**:
- Parameters: 793,473
- Size: 3.02 MB
- Input: 768 features (12 bitboards, re-ordered by side-to-move)
- Designed for highest accuracy on larger GPUs

---

**Need more details?** See:
- `READY_TO_TRAIN.md` - Complete setup guide
- `CHECKPOINT_GUIDE.md` - Resumption documentation
- `README_RTX5070.md` - GPU optimization guide
