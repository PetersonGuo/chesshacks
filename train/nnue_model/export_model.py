"""
Export NNUE model weights to binary format for C++ inference

This script exports a trained PyTorch NNUE model to a custom binary format
that can be loaded and used for fast inference in C++.

Binary format:
- Magic number: "NNUE" (4 bytes)
- Version: uint32 (4 bytes)
- Hidden sizes: 3 x uint32 (12 bytes total)
- Layer weights and biases in order:
  - ft.weight: [hidden_size, 768] (row-major)
  - ft.bias: [hidden_size]
  - fc1.weight: [hidden2_size, hidden_size] (row-major)
  - fc1.bias: [hidden2_size]
  - fc2.weight: [hidden3_size, hidden2_size] (row-major)
  - fc2.bias: [hidden3_size]
  - fc3.weight: [1, hidden3_size] (row-major)
  - fc3.bias: [1]

All weights are stored as float32 in little-endian format.
"""

import torch
import struct
import sys
import os
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent))
from model import ChessNNUEModel


def export_model_to_binary(model_path: str, output_path: str):
    """
    Export NNUE model to binary format for C++

    Args:
        model_path: Path to PyTorch model (.pt file)
        output_path: Path to output binary file
    """
    print(f"Loading model from {model_path}...")

    # Load the model
    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded checkpoint from epoch {epoch}")
        else:
            state_dict = checkpoint
    else:
        # Assume it's directly the state dict or model
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

    # Handle models with _orig_mod. prefix (from torch.compile)
    # Remove the prefix if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Detected _orig_mod. prefix (from torch.compile), removing...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Create model and load weights
    # Try to infer hidden sizes from state dict
    ft_friendly_weight_shape = state_dict['ft_friendly.weight'].shape
    fc1_weight_shape = state_dict['fc1.weight'].shape
    fc2_weight_shape = state_dict['fc2.weight'].shape

    hidden_size = ft_friendly_weight_shape[0]  # Output size of ft_friendly layer
    hidden2_size = fc1_weight_shape[0]  # Output size of fc1 layer
    hidden3_size = fc2_weight_shape[0]  # Output size of fc2 layer

    print(f"Model architecture:")
    print(f"  Input size: 768 (384+384 bitboard features)")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Hidden2 size: {hidden2_size}")
    print(f"  Hidden3 size: {hidden3_size}")
    print(f"  Output size: 1")

    # Create model instance
    model = ChessNNUEModel(
        hidden_size=hidden_size,
        hidden2_size=hidden2_size,
        hidden3_size=hidden3_size
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\nExporting to {output_path}...")

    # Open output file in binary write mode
    with open(output_path, 'wb') as f:
        # Write magic number
        f.write(b'NNUE')

        # Write version 2 (dual feature transformers + residual)
        f.write(struct.pack('<I', 2))  # Version 2

        # Write architecture (hidden sizes)
        f.write(struct.pack('<I', hidden_size))
        f.write(struct.pack('<I', hidden2_size))
        f.write(struct.pack('<I', hidden3_size))

        # Helper function to write layer weights
        def write_layer(name, weight_key, bias_key):
            weight = state_dict[weight_key].cpu().numpy()
            bias = state_dict[bias_key].cpu().numpy()

            print(f"  Writing {name}: weight shape {weight.shape}, bias shape {bias.shape}")

            # Write weights (row-major, which is the default for numpy)
            f.write(weight.astype('float32').tobytes())

            # Write biases
            f.write(bias.astype('float32').tobytes())

        # Write all layers in order (dual FT + residual architecture)
        write_layer("Feature Transformer Friendly (ft_friendly)", "ft_friendly.weight", "ft_friendly.bias")
        write_layer("Feature Transformer Enemy (ft_enemy)", "ft_enemy.weight", "ft_enemy.bias")
        write_layer("Hidden Layer 1 (fc1)", "fc1.weight", "fc1.bias")
        write_layer("Residual Layer 1 (res1)", "res1.weight", "res1.bias")
        write_layer("Residual Layer 2 (res2)", "res2.weight", "res2.bias")
        write_layer("Hidden Layer 2 (fc2)", "fc2.weight", "fc2.bias")
        write_layer("Output Layer (fc3)", "fc3.weight", "fc3.bias")

    file_size = os.path.getsize(output_path)
    print(f"\nExport complete! File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

    # Verify the export by checking file size
    expected_params = (
        384 * hidden_size + hidden_size +  # ft_friendly
        384 * hidden_size + hidden_size +  # ft_enemy
        hidden_size * hidden2_size + hidden2_size +  # fc1
        hidden2_size * hidden2_size + hidden2_size +  # res1
        hidden2_size * hidden2_size + hidden2_size +  # res2
        hidden2_size * hidden3_size + hidden3_size +  # fc2
        hidden3_size * 1 + 1  # fc3
    )
    expected_size = 4 + 4 + 12 + expected_params * 4  # magic + version + arch + params

    if file_size == expected_size:
        print(f"✓ File size matches expected size ({expected_size:,} bytes)")
    else:
        print(f"⚠ Warning: File size mismatch! Expected {expected_size:,}, got {file_size:,}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export NNUE model to binary format for C++')
    parser.add_argument('model_path', type=str, help='Path to PyTorch model (.pt file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for binary file (default: model_path with .bin extension)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        output_path = args.model_path.replace('.pt', '.bin')
    else:
        output_path = args.output

    # Export model
    export_model_to_binary(args.model_path, output_path)


if __name__ == "__main__":
    main()
