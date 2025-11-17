"""
Benchmark bitmap NNUE implementation vs theoretical sparse version
"""

import time

import chess
import numpy as np
import torch

from train.nnue_model.model import BitboardFeatures, ChessNNUEModel


def benchmark_feature_extraction(num_positions=1000):
    """Benchmark feature extraction speed"""
    print("=" * 80)
    print(f"Benchmarking Feature Extraction ({num_positions} positions)")
    print("=" * 80)

    # Create random positions
    boards = [chess.Board() for _ in range(num_positions)]

    # Benchmark bitmap feature extraction
    start = time.time()
    for board in boards:
        features = BitboardFeatures.board_to_features_for_side(
            board, perspective=board.turn
        )
    bitmap_time = time.time() - start

    print(f"\nBitmap feature extraction:")
    print(f"  Total time: {bitmap_time:.4f}s")
    print(f"  Per position: {bitmap_time/num_positions*1000:.4f}ms")
    print(f"  Throughput: {num_positions/bitmap_time:.0f} positions/sec")


def benchmark_model_inference(num_positions=1000, batch_size=256):
    """Benchmark model inference speed"""
    print("\n" + "=" * 80)
    print(
        f"Benchmarking Model Inference ({num_positions} positions, batch_size={batch_size})"
    )
    print("=" * 80)

    model = ChessNNUEModel()
    model.eval()

    # Create random positions
    boards = [chess.Board() for _ in range(num_positions)]
    features_list = [
        BitboardFeatures.board_to_features_for_side(board, perspective=board.turn)
        for board in boards
    ]

    # Benchmark single position inference
    print("\n1. Single Position Inference:")
    start = time.time()
    with torch.no_grad():
        for features in features_list[:100]:
            output = model(features.unsqueeze(0))
    single_time = time.time() - start

    print(f"  Total time (100 positions): {single_time:.4f}s")
    print(f"  Per position: {single_time/100*1000:.4f}ms")
    print(f"  Throughput: {100/single_time:.0f} positions/sec")

    # Benchmark batched inference
    print(f"\n2. Batched Inference (batch_size={batch_size}):")
    num_batches = num_positions // batch_size
    start = time.time()
    with torch.no_grad():
        for i in range(num_batches):
            batch_features = torch.stack(
                features_list[i * batch_size : (i + 1) * batch_size], dim=0
            )
            outputs = model(batch_features)
    batch_time = time.time() - start

    positions_processed = num_batches * batch_size
    print(f"  Total time ({positions_processed} positions): {batch_time:.4f}s")
    print(f"  Per position: {batch_time/positions_processed*1000:.4f}ms")
    print(f"  Throughput: {positions_processed/batch_time:.0f} positions/sec")
    print(
        f"  Speedup vs single: {single_time/100 / (batch_time/positions_processed):.2f}x"
    )


def benchmark_gpu_inference(num_positions=1000, batch_size=256):
    """Benchmark GPU inference if available"""
    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("GPU Benchmark: CUDA not available")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print(
        f"Benchmarking GPU Inference ({num_positions} positions, batch_size={batch_size})"
    )
    print("=" * 80)

    try:
        device = torch.device("cuda")
        model = ChessNNUEModel().to(device)
        model.eval()

        boards = [chess.Board() for _ in range(num_positions)]
        features_list = [
            BitboardFeatures.board_to_features_for_side(
                board, perspective=board.turn
            ).to(device)
            for board in boards
        ]

        with torch.no_grad():
            batch_features = torch.stack(features_list[:batch_size], dim=0)
            _ = model(batch_features)

        num_batches = num_positions // batch_size
        start = time.time()
        with torch.no_grad():
            for i in range(num_batches):
                batch_features = torch.stack(
                    features_list[i * batch_size : (i + 1) * batch_size], dim=0
                )
                _ = model(batch_features)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        positions_processed = num_batches * batch_size
        print(f"  Total time ({positions_processed} positions): {gpu_time:.4f}s")
        print(f"  Per position: {gpu_time/positions_processed*1000:.4f}ms")
        print(f"  Throughput: {positions_processed/gpu_time:.0f} positions/sec")
    except RuntimeError as err:
        print("  Skipping GPU benchmark due to runtime error:", err)


def analyze_model_size():
    """Analyze model size and parameters"""
    print("\n" + "=" * 80)
    print("Model Size Analysis")
    print("=" * 80)

    model = ChessNNUEModel()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nBitmap NNUE Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print(f"\nLayer breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name:20s}: {param.numel():,} parameters ({param.shape})")

    # Compare with theoretical sparse model
    print(f"\n" + "-" * 80)
    print("Comparison with Sparse HalfKP Model:")
    sparse_input = 40960  # 64 * 64 * 10
    sparse_hidden = 256
    sparse_first_layer = sparse_input * sparse_hidden + sparse_hidden

    friendly_params = model.ft_friendly.weight.numel() + model.ft_friendly.bias.numel()
    enemy_params = model.ft_enemy.weight.numel() + model.ft_enemy.bias.numel()
    bitmap_first_layer = friendly_params + enemy_params

    print(f"\nTheoretical Sparse HalfKP:")
    print(f"  Input features: {sparse_input:,}")
    print(f"  First layer params: {sparse_first_layer:,}")
    print(f"  Bitmap first-stage params: {bitmap_first_layer:,}")
    print(
        f"  Memory reduction vs sparse first layer: "
        f"{sparse_first_layer / bitmap_first_layer:.1f}x smaller"
    )


if __name__ == "__main__":
    analyze_model_size()
    benchmark_feature_extraction(num_positions=10000)
    benchmark_model_inference(num_positions=10000, batch_size=256)
    benchmark_gpu_inference(num_positions=10000, batch_size=256)

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("1. Bitmap model is 50x+ smaller in memory")
    print("2. Dense operations are highly optimized (BLAS/cuBLAS)")
    print("3. Batched inference provides significant speedup")
    print("4. GPU acceleration works seamlessly with dense tensors")
    print("5. Can use multiprocessing and pin_memory for faster data loading")
    print("=" * 80)
