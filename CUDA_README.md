# CUDA GPU Acceleration

This document describes the CUDA GPU acceleration implementation in ChessHacks.

## Overview

ChessHacks now includes CUDA support for GPU-accelerated chess position evaluation. This enables batch evaluation of chess positions on NVIDIA GPUs, which can significantly speed up search for positions with many child nodes.

## Features

- **Batch Position Evaluation**: Evaluate hundreds of chess positions simultaneously on GPU
- **Automatic Fallback**: Gracefully falls back to CPU if CUDA is not available
- **Conditional Compilation**: Builds with or without CUDA depending on availability
- **GPU Memory Optimization**: Uses CUDA constant memory for piece-square tables (faster access)

## Requirements

### Hardware

- NVIDIA GPU with CUDA Compute Capability 7.5 or higher
  - Turing (RTX 20 series, GTX 16 series): Compute 7.5
  - Ampere (RTX 30 series): Compute 8.0, 8.6
  - Ada Lovelace (RTX 40 series): Compute 8.9
  - Blackwell (RTX 50 series): Compute 12.0

### Software

- CUDA Toolkit 11.0 or higher (tested with 12.3)
- CMake 3.18 or higher
- NVIDIA GPU drivers

## Building with CUDA

The build system automatically detects CUDA and enables GPU acceleration if available:

```bash
cd chesshacks
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

If CUDA is found, you'll see:

```
-- CUDA found - enabling GPU acceleration
-- Found CUDAToolkit: /usr/local/cuda-X.X/include (found version "X.X.XX")
```

If CUDA is not found:

```
-- CUDA not found - GPU acceleration disabled
```

## Usage

### Checking CUDA Availability

```python
import c_helpers

# Check if CUDA is available
if c_helpers.is_cuda_available():
    print("CUDA is available!")
    print(c_helpers.get_cuda_info())
else:
    print("CUDA not available - using CPU")
```

### Using CUDA-Accelerated Search

```python
import c_helpers

# Create search data structures
tt = c_helpers.TranspositionTable()
killers = c_helpers.KillerMoves()
history = c_helpers.HistoryTable()

# Evaluation function
def evaluate(fen):
    return c_helpers.evaluate_with_pst(fen)

# CUDA-accelerated search
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
score = c_helpers.alpha_beta_cuda(
    fen, depth=5,
    alpha=c_helpers.MIN,
    beta=c_helpers.MAX,
    maximizingPlayer=True,
    evaluate=evaluate,
    tt=tt,
    killers=killers,
    history=history
)
```

## Implementation Details

### CUDA Kernels

The implementation uses several CUDA features for optimal performance:

1. **Constant Memory**: Piece-square tables are stored in GPU constant memory, which provides:

   - Cached reads (faster than global memory)
   - Broadcast to all threads in a warp
   - Read-only data with good locality

2. **Batch Evaluation Kernel**:

   - Processes 256 positions per thread block
   - Each thread evaluates one position independently
   - Material counting and piece-square table lookups parallelized

3. **Memory Management**:
   - Automatic GPU memory allocation/deallocation
   - Host-to-device memory transfers batched for efficiency
   - Device-to-host result transfers

### File Structure

```
src/
├── cuda/                    # CUDA GPU acceleration (separate folder)
│   ├── cuda_eval.cu         # CUDA kernels for batch evaluation
│   ├── cuda_eval.h          # CUDA evaluation interface
│   ├── cuda_utils.cpp       # CUDA detection and info
│   └── cuda_utils.h         # CUDA utility headers
├── chess_board.cpp
├── evaluation.cpp
├── search_and_utils.cpp     # Alpha-beta with CUDA integration
└── ...
```

### Performance Characteristics

The CUDA implementation is most beneficial when:

- Evaluating many positions (batch size > 100)
- Deep searches with high branching factor
- Positions with many legal moves

For shallow searches or positions with few moves, CPU overhead may dominate.

## Benchmarking

Run the included benchmark script to compare CPU vs GPU performance:

```bash
python3 benchmark_cuda.py
```

Example output:

```
============================================================
ChessHacks CPU vs CUDA Performance Comparison
============================================================

CUDA Available: True
GPU: NVIDIA GeForce RTX 5070 (CUDA Compute 12.0)

Test Position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Average times by depth:
Depth      CPU (s)         CUDA (s)        Speedup
------------------------------------------------------------
3          0.0034          0.0033          1.01x
4          0.0167          0.0165          1.01x
5          0.1608          0.1605          1.00x

Overall average speedup: 1.01x
```

## Testing

Run the CUDA test suite:

```bash
python3 test_cuda.py
```

This tests:

1. CUDA detection and GPU info
2. Basic position evaluation
3. Alpha-beta search with CUDA

## Troubleshooting

### CUDA not detected

If CUDA is installed but not detected:

```bash
# Check CUDA compiler
which nvcc

# Check CUDA version
nvcc --version

# Verify GPU
nvidia-smi

# Set CUDA path (if needed)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then rebuild:

```bash
cd build
rm -rf *
cmake ..
make -j$(nproc)
```

### Runtime CUDA errors

Check GPU memory:

```bash
nvidia-smi
```

Verify compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Update `CMakeLists.txt` if your GPU has a different compute capability:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # Add your compute capability
```

## Future Improvements

Potential enhancements for the CUDA implementation:

1. **Full Search on GPU**: Move entire alpha-beta tree traversal to GPU
2. **Shared Memory Optimization**: Use shared memory for transposition table
3. **Multi-GPU Support**: Distribute search across multiple GPUs
4. **Asynchronous Evaluation**: Overlap CPU search with GPU evaluation
5. **Move Generation on GPU**: Generate legal moves on GPU

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Chess Programming Wiki - Parallel Search](https://www.chessprogramming.org/Parallel_Search)

## License

Same license as ChessHacks main project.
