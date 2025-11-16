# NNUE Model Integration Guide

This guide explains how to train, export, and use NNUE models for fast chess position evaluation in the engine.

## Overview

The engine now supports NNUE (Efficiently Updatable Neural Network) evaluation, which provides:
- **Stronger positional understanding** than piece-square tables
- **Fast inference** (optimized C++ implementation)
- **No external dependencies** (custom binary format, no PyTorch/LibTorch required at runtime)
- **Automatic fallback** to PST evaluation if NNUE not loaded

## Architecture

**Model:** Virgo-style bitboard NNUE
- **Input:** 768 features (2 colors × 6 piece types × 64 squares)
- **Architecture:** 768 → 256 → 32 → 32 → 1
- **Activations:** ClippedReLU for hidden layers, linear for output
- **Output:** Evaluation score in centipawns (positive = white advantage)

## Quick Start

### 1. Train an NNUE Model

First, you need to train an NNUE model using the training pipeline:

```bash
# Navigate to the NNUE training directory
cd train/nnue_model

# Train a model (this will download data if needed)
python train.py --config rtx5070  # Or use 'default', 'fast', etc.
```

The trained model will be saved to `train/nnue_model/checkpoints/`.

For more details on training, see `train/nnue_model/READY_TO_TRAIN.md`.

### 2. Export Model to Binary Format

Convert the PyTorch model to a binary format for C++ inference:

```bash
# Export the best model
python train/nnue_model/export_model.py train/nnue_model/checkpoints/best_model.pt

# This creates: train/nnue_model/checkpoints/best_model.bin
```

### 3. Build the Engine with NNUE Support

Build the C++ extension:

```bash
mkdir -p build
cd build
cmake ..
make -j4
```

The NNUE evaluator is automatically included in the build.

### 4. Use NNUE in Your Code

#### Python Usage

```python
import sys
sys.path.insert(0, 'build')
import c_helpers

# Initialize NNUE evaluator
model_path = "train/nnue_model/checkpoints/best_model.bin"
success = c_helpers.init_nnue(model_path)

if success:
    print("NNUE loaded successfully!")
else:
    print("Failed to load NNUE, will use PST evaluation")

# Evaluate positions
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Option 1: Use evaluate() - automatically uses NNUE if loaded
score = c_helpers.evaluate(fen)
print(f"Evaluation: {score} centipawns")

# Option 2: Explicitly use NNUE (falls back to PST if not loaded)
nnue_score = c_helpers.evaluate_nnue(fen)

# Option 3: Always use PST (ignore NNUE)
pst_score = c_helpers.evaluate_with_pst(fen)

# Check if NNUE is loaded
if c_helpers.is_nnue_loaded():
    print("NNUE is active")
```

#### C++ Usage

```cpp
#include "evaluation.h"

int main() {
    // Initialize NNUE
    std::string model_path = "train/nnue_model/checkpoints/best_model.bin";
    bool success = init_nnue(model_path);

    if (success) {
        std::cout << "NNUE loaded successfully!" << std::endl;
    }

    // Evaluate positions
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Option 1: Use evaluate() - automatically uses NNUE if loaded
    int score = evaluate(fen);

    // Option 2: Explicitly use NNUE
    int nnue_score = evaluate_nnue(fen);

    // Option 3: Always use PST
    int pst_score = evaluate_with_pst(fen);

    // Check if NNUE is loaded
    if (is_nnue_loaded()) {
        std::cout << "NNUE is active" << std::endl;
    }

    return 0;
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_nnue.py
```

This will:
1. Find a trained NNUE model
2. Export it to binary format
3. Build the C++ extension
4. Test NNUE evaluation on various positions

## File Structure

```
chesshacks/
├── src/
│   ├── nnue_evaluator.h         # NNUE evaluator class header
│   ├── nnue_evaluator.cpp       # NNUE evaluator implementation
│   ├── evaluation.h             # Updated with NNUE functions
│   ├── evaluation.cpp           # Integrated NNUE evaluation
│   └── bindings.cpp             # Python bindings for NNUE
├── train/
│   └── nnue_model/
│       ├── model.py             # NNUE model architecture (PyTorch)
│       ├── train.py             # Training script
│       ├── export_model.py      # Export to binary format
│       └── checkpoints/         # Saved models (.pt and .bin)
├── test_nnue.py                 # NNUE test script
└── NNUE_USAGE.md               # This file
```

## Binary Format Specification

The exported NNUE model uses a custom binary format:

```
Header:
- Magic number: "NNUE" (4 bytes)
- Version: uint32 (4 bytes) - currently 1
- Hidden sizes: 3 × uint32 (12 bytes)
  - hidden_size (feature transformer output)
  - hidden2_size (first hidden layer output)
  - hidden3_size (second hidden layer output)

Weights (all float32, little-endian):
- ft.weight: [hidden_size, 768]
- ft.bias: [hidden_size]
- fc1.weight: [hidden2_size, hidden_size]
- fc1.bias: [hidden2_size]
- fc2.weight: [hidden3_size, hidden2_size]
- fc2.bias: [hidden3_size]
- fc3.weight: [1, hidden3_size]
- fc3.bias: [1]
```

## Performance

The NNUE evaluator is optimized for fast inference:
- **No external dependencies** at runtime (no PyTorch/LibTorch)
- **Efficient matrix operations** using native C++
- **Cache-friendly memory layout**
- **Typical inference time:** < 0.1ms per position (CPU)

For batch evaluation, you can still use `batch_evaluate_mt()` which will use NNUE if loaded.

## Troubleshooting

### "NNUE model not loaded" error
- Make sure you've called `init_nnue()` with the correct path
- Check that the .bin file exists and is readable
- Verify the model was exported correctly (run `export_model.py`)

### Wrong evaluations
- Ensure you're using an NNUE model (Virgo-style bitboard), not a CNN model
- Check that the model was trained with the correct architecture (768 → 256 → 32 → 32 → 1)
- Verify the normalization scheme matches (centipawns / 100)

### Build errors
- Make sure all source files are added to CMakeLists.txt
- Check that you have C++14 or later
- Ensure nanobind is installed: `pip install nanobind`

## Advanced Usage

### Custom Model Architectures

To use a different model architecture:

1. Update `train/nnue_model/model.py` with your architecture
2. Train the model
3. Update `NNUEEvaluator` class in `src/nnue_evaluator.h/cpp` to match
4. Export and test

### Integration with Search

The NNUE evaluator automatically integrates with the search functions:

```python
# The search will use NNUE evaluation if loaded
c_helpers.init_nnue("path/to/model.bin")

# All search functions now use NNUE
best_move = c_helpers.find_best_move(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    depth=6,
    evaluate=c_helpers.evaluate  # Uses NNUE if loaded
)
```

## References

- [NNUE Paper](https://arxiv.org/abs/2007.03574) - Original NNUE architecture
- [Stockfish NNUE](https://github.com/official-stockfish/nnue-pytorch) - Inspiration for implementation
- [Virgo Engine](https://github.com/alex-unofficial/virgo) - Bitboard representation used

## Contributing

Improvements to the NNUE implementation are welcome! Areas for future work:
- [ ] Incremental update (efficiently updatable features)
- [ ] Larger model architectures
- [ ] Quantization (int8/int16) for faster inference
- [ ] SIMD optimizations (AVX2/AVX512)
- [ ] GPU inference support
