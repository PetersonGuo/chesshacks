#!/usr/bin/env python3
"""
Simple demonstration of NNUE integration

This script shows how to use the NNUE evaluator once you have a trained model.
"""

import sys
from pathlib import Path

# Add build directory to path
build_dir = Path(__file__).parent / "build"
sys.path.insert(0, str(build_dir))

import c_helpers

def main():
    print("="*80)
    print("NNUE Integration Demo")
    print("="*80)

    # Test positions
    positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "After 1.e4 e5"),
    ]

    print("\n1. Testing PST Evaluation (always available)")
    print("-" * 80)
    print(f"{'Position':<30} {'Evaluation':>15}")
    print("-" * 80)

    for fen, desc in positions:
        score = c_helpers.evaluate_with_pst(fen)
        print(f"{desc:<30} {score:>15} cp")

    print("\n2. Testing NNUE Integration")
    print("-" * 80)

    # Check if NNUE is loaded
    if c_helpers.is_nnue_loaded():
        print("✓ NNUE model is loaded")
        print("\nNNUE Evaluations:")
        print(f"{'Position':<30} {'Evaluation':>15}")
        print("-" * 80)

        for fen, desc in positions:
            score = c_helpers.evaluate_nnue(fen)
            print(f"{desc:<30} {score:>15} cp")

    else:
        print("✗ NNUE model not loaded")
        print("\nTo use NNUE:")
        print("1. Train a model: python train/nnue_model/train.py")
        print("2. Export to binary: python train/nnue_model/export_model.py <model.pt>")
        print("3. Load in code: c_helpers.init_nnue('path/to/model.bin')")

    print("\n3. Testing evaluate() function (uses NNUE if loaded, else PST)")
    print("-" * 80)
    print(f"{'Position':<30} {'Evaluation':>15}")
    print("-" * 80)

    for fen, desc in positions:
        score = c_helpers.evaluate(fen)
        print(f"{desc:<30} {score:>15} cp")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)

    # Print usage example
    print("\nExample usage in your code:")
    print("-" * 80)
    print("""
# Option 1: Use NNUE if you have a trained model
import c_helpers

# Load NNUE model
if c_helpers.init_nnue("train/nnue_model/checkpoints/model.bin"):
    print("NNUE loaded!")

# Evaluate positions (automatically uses NNUE if loaded)
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
score = c_helpers.evaluate(fen)
print(f"Evaluation: {score} cp")

# Option 2: Use PST evaluation (no NNUE needed)
score = c_helpers.evaluate_with_pst(fen)
print(f"PST Evaluation: {score} cp")
    """)


if __name__ == "__main__":
    main()
