"""
Test the complete training pipeline with bitmap NNUE model
Verifies RTX 5070 configuration compatibility
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import tempfile

import chess
import pytest
import torch
from torch.utils.data import DataLoader

from train.nnue_model.model import ChessNNUEModel, BitboardFeatures
from train.nnue_model.train import ChessBoardDatasetBitmap, collate_fn_bitmap, create_dataloaders
from train.config import get_config


def create_test_data(num_positions=1000):
    """Create temporary test dataset"""
    positions = []

    # Generate random positions with evaluations
    for i in range(num_positions):
        board = chess.Board()

        # Make a few random moves
        for _ in range(min(i % 20, len(list(board.legal_moves)))):
            moves = list(board.legal_moves)
            if moves:
                board.push(moves[i % len(moves)])

        # Fake evaluation (in practice, use Stockfish)
        eval_score = (i % 200) - 100  # Range from -100 to 100

        positions.append({
            'fen': board.fen(),
            'eval': eval_score
        })

    return positions


def _write_positions(path, positions):
    with open(path, 'w') as f:
        for pos in positions:
            f.write(json.dumps(pos) + '\n')


@pytest.fixture(scope="module")
def temp_file(tmp_path_factory):
    """Shared temporary dataset file for unit tests"""
    data_dir = tmp_path_factory.mktemp("nnue_bitmap")
    data_path = data_dir / "positions.jsonl"
    _write_positions(data_path, create_test_data(100))
    return str(data_path)


def create_temp_dataset_file(num_positions=100):
    """Helper for running this module as a standalone script"""
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    _write_positions(path, create_test_data(num_positions))
    return path


def test_dataset(temp_file):
    """Test bitmap dataset creation"""
    print("=" * 80)
    print("Testing Bitmap Dataset")
    print("=" * 80)
    # Create dataset
    dataset = ChessBoardDatasetBitmap(
        data_path=temp_file,
        max_positions=100,
        normalize=True
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Normalization: mean={dataset.eval_mean:.4f}, std={dataset.eval_std:.4f}")

    # Test data loading
    features, eval_score = dataset[0]
    print(f"\nSample data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Features dtype: {features.dtype}")
    print(f"  Evaluation: {eval_score:.4f}")
    print(f"  Non-zero features: {(features != 0).sum().item()}")

    print("\n✓ Dataset test passed!")


def test_dataloader(temp_file):
    """Test bitmap dataloader"""
    print("\n" + "=" * 80)
    print("Testing Bitmap DataLoader")
    print("=" * 80)

    # Create dataset
    dataset = ChessBoardDatasetBitmap(
        data_path=temp_file,
        max_positions=100,
        normalize=True
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        collate_fn=collate_fn_bitmap
    )

    # Test batch loading
    for features, targets in loader:
        print(f"\nBatch loaded:")
        print(f"  Features shape: {features.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Features dtype: {features.dtype}")
        print(f"  Targets dtype: {targets.dtype}")
        break

    print("\n✓ DataLoader test passed!")


def test_model_training(temp_file):
    """Test model training on sample data"""
    print("\n" + "=" * 80)
    print("Testing Model Training")
    print("=" * 80)

    # Create model
    model = ChessNNUEModel(hidden_size=256, hidden2_size=32, hidden3_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset
    dataset = ChessBoardDatasetBitmap(
        data_path=temp_file,
        max_positions=100,
        normalize=True
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_bitmap,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Train for 1 epoch
    model.train()
    total_loss = 0.0
    num_batches = 0

    print("\nTraining for 1 epoch...")
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Average loss: {avg_loss:.4f}")
    print("\n✓ Training test passed!")


def test_rtx5070_config():
    """Test RTX 5070 configuration"""
    print("\n" + "=" * 80)
    print("Testing RTX 5070 Configuration")
    print("=" * 80)

    # Load RTX 5070 config
    config = get_config('rtx5070')

    print("\nRTX 5070 Standard Config:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Num workers: {config.num_workers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num epochs: {config.num_epochs}")

    # Load RTX 5070 quality config
    quality_config = get_config('rtx5070_quality')

    print("\nRTX 5070 Quality Config:")
    print(f"  Device: {quality_config.device}")
    print(f"  Batch size: {quality_config.batch_size}")
    print(f"  Learning rate: {quality_config.learning_rate}")
    print(f"  Num workers: {quality_config.num_workers}")
    print(f"  Hidden size: {quality_config.hidden_size}")
    print(f"  Num epochs: {quality_config.num_epochs}")

    # Create models with both configs
    model_standard = ChessNNUEModel(
        hidden_size=config.hidden_size,
        hidden2_size=config.hidden2_size,
        hidden3_size=config.hidden3_size
    )

    model_quality = ChessNNUEModel(
        hidden_size=quality_config.hidden_size,
        hidden2_size=quality_config.hidden2_size,
        hidden3_size=quality_config.hidden3_size
    )

    print(f"\nStandard model parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
    print(f"Quality model parameters: {sum(p.numel() for p in model_quality.parameters()):,}")

    print("\n✓ RTX 5070 config test passed!")


def test_end_to_end():
    """Test complete end-to-end pipeline"""
    print("\n" + "=" * 80)
    print("Testing Complete Pipeline")
    print("=" * 80)

    # Create temporary train/val data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        train_file = f.name
        train_data = create_test_data(500)
        for pos in train_data:
            f.write(json.dumps(pos) + '\n')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        val_file = f.name
        val_data = create_test_data(100)
        for pos in val_data:
            f.write(json.dumps(pos) + '\n')

    try:
        # Create dataloaders using the pipeline function
        train_loader, val_loader = create_dataloaders(
            train_path=train_file,
            val_path=val_file,
            batch_size=32,
            num_workers=0,
            normalize=True
        )

        print(f"\nDataloaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # Test loading
        for features, targets in train_loader:
            print(f"\nTrain batch:")
            print(f"  Features: {features.shape}")
            print(f"  Targets: {targets.shape}")
            break

        for features, targets in val_loader:
            print(f"\nVal batch:")
            print(f"  Features: {features.shape}")
            print(f"  Targets: {targets.shape}")
            break

        print("\n✓ End-to-end pipeline test passed!")

    finally:
        os.unlink(train_file)
        os.unlink(val_file)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BITMAP NNUE TRAINING PIPELINE TESTS")
    print("=" * 80)

    try:
        # Run all tests
        temp_file = create_temp_dataset_file(100)
        test_dataset(temp_file)
        test_dataloader(temp_file)
        test_model_training(temp_file)
        test_rtx5070_config()
        test_end_to_end()

        # Cleanup
        os.unlink(temp_file)

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nThe bitmap NNUE model is fully compatible with:")
        print("  • Bitmap/bitboard feature representation")
        print("  • RTX 5070 optimized configurations")
        print("  • Complete training pipeline")
        print("  • GPU acceleration (when available)")
        print("\nReady for training!")

    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
