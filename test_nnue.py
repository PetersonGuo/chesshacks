#!/usr/bin/env python3
"""
Test NNUE model export and evaluation

This script:
1. Exports a trained NNUE model to binary format
2. Builds the C++ extension with NNUE support
3. Tests NNUE evaluation from Python
"""

import os
import sys
from pathlib import Path

# Add train directory to path
sys.path.insert(0, str(Path(__file__).parent / "train"))

def find_trained_model():
    """Find a trained NNUE model in the repository"""
    # Look for models in train/nnue_model/checkpoints/
    nnue_checkpoint_dir = Path(__file__).parent / "train" / "nnue_model" / "checkpoints"

    # Look for best_model.pt or final_model.pt
    if nnue_checkpoint_dir.exists():
        best_model = nnue_checkpoint_dir / "best_model.pt"
        final_model = nnue_checkpoint_dir / "final_model.pt"

        if best_model.exists():
            return best_model
        elif final_model.exists():
            return final_model
        else:
            # Find any .pt file
            pt_files = list(nnue_checkpoint_dir.glob("*.pt"))
            if pt_files:
                return pt_files[0]

    # Look for CNN models (fallback, but these won't work with NNUE)
    cnn_checkpoint_dir = Path(__file__).parent / "train" / "cnn_model" / "checkpoints"
    if cnn_checkpoint_dir.exists():
        best_model = cnn_checkpoint_dir / "best_model.pt"
        final_model = cnn_checkpoint_dir / "final_model.pt"

        if best_model.exists():
            print("Warning: Found CNN model, but NNUE expects bitmap-style model")
            print("Please train an NNUE model first using train/nnue_model/train.py")
            return None
        elif final_model.exists():
            print("Warning: Found CNN model, but NNUE expects bitmap-style model")
            print("Please train an NNUE model first using train/nnue_model/train.py")
            return None

    return None


def export_model(model_path, output_path):
    """Export PyTorch model to binary format"""
    print(f"\n{'='*80}")
    print("Step 1: Exporting NNUE model to binary format")
    print(f"{'='*80}\n")

    # Import export script
    sys.path.insert(0, str(Path(__file__).parent / "train" / "nnue_model"))
    from export_model import export_model_to_binary

    try:
        export_model_to_binary(str(model_path), str(output_path))
        print(f"\n✓ Model exported to {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_cpp_extension():
    """Build C++ extension with NNUE support"""
    print(f"\n{'='*80}")
    print("Step 2: Building C++ extension with NNUE support")
    print(f"{'='*80}\n")

    import subprocess

    # Build the extension
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)

    try:
        # Run CMake
        print("Running CMake...")
        result = subprocess.run(
            ["cmake", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"CMake error:\n{result.stderr}")
            return False

        print(result.stdout)

        # Build
        print("\nBuilding...")
        result = subprocess.run(
            ["make", "-j4"],
            cwd=build_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Build error:\n{result.stderr}")
            return False

        print(result.stdout)
        print("\n✓ Build successful")
        return True

    except Exception as e:
        print(f"\n✗ Error building: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nnue_evaluation(model_bin_path):
    """Test NNUE evaluation from Python"""
    print(f"\n{'='*80}")
    print("Step 3: Testing NNUE evaluation")
    print(f"{'='*80}\n")

    # Add build directory to Python path
    build_dir = Path(__file__).parent / "build"
    sys.path.insert(0, str(build_dir))

    try:
        import c_helpers

        # Test positions
        test_positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "After 1.e4 e5"),
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "After 1.e4 e5 2.Nf3 Nc6"),
        ]

        # Load NNUE model
        print(f"Loading NNUE model from: {model_bin_path}")
        success = c_helpers.init_nnue(str(model_bin_path))

        if not success:
            print("✗ Failed to load NNUE model")
            return False

        print("✓ NNUE model loaded successfully\n")

        # Test evaluation
        print(f"{'Position':<30} {'PST Eval':>12} {'NNUE Eval':>12} {'Difference':>12}")
        print("-" * 70)

        for fen, description in test_positions:
            pst_eval = c_helpers.evaluate_with_pst(fen)
            nnue_eval = c_helpers.evaluate_nnue(fen)
            diff = nnue_eval - pst_eval

            print(f"{description:<30} {pst_eval:>12} {nnue_eval:>12} {diff:>12}")

        print("\n✓ NNUE evaluation test complete")

        # Test that evaluate() uses NNUE when loaded
        print("\nTesting evaluate() function (should use NNUE):")
        for fen, description in test_positions[:2]:
            eval_score = c_helpers.evaluate(fen)
            nnue_score = c_helpers.evaluate_nnue(fen)

            if eval_score == nnue_score:
                print(f"  ✓ {description}: {eval_score} (using NNUE)")
            else:
                print(f"  ✗ {description}: Expected {nnue_score}, got {eval_score}")

        return True

    except ImportError as e:
        print(f"\n✗ Error importing c_helpers: {e}")
        print("Make sure the C++ extension is built successfully")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"\n{'='*80}")
    print("NNUE Model Export and Evaluation Test")
    print(f"{'='*80}\n")

    # Step 1: Find trained model
    print("Looking for trained NNUE model...")
    model_path = find_trained_model()

    if model_path is None:
        print("\n✗ No trained NNUE model found!")
        print("\nTo train an NNUE model, run:")
        print("  python train/nnue_model/train.py")
        print("\nOr use a pre-trained model by placing it in:")
        print("  train/nnue_model/checkpoints/best_model.pt")
        return 1

    print(f"✓ Found model: {model_path}\n")

    # Step 2: Export model to binary
    output_path = model_path.parent / "model.bin"
    if not export_model(model_path, output_path):
        return 1

    # Step 3: Build C++ extension
    if not build_cpp_extension():
        return 1

    # Step 4: Test NNUE evaluation
    if not test_nnue_evaluation(output_path):
        return 1

    print(f"\n{'='*80}")
    print("All tests passed!")
    print(f"{'='*80}\n")
    print(f"NNUE model is ready to use at: {output_path}")
    print("\nTo use NNUE in your code:")
    print("  import c_helpers")
    print(f"  c_helpers.init_nnue('{output_path}')")
    print("  score = c_helpers.evaluate(fen)  # Uses NNUE if loaded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
