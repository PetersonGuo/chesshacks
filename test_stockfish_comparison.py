#!/usr/bin/env python3
"""
Test script to demonstrate Stockfish comparison functionality.

This script shows how to enable Stockfish comparison and view statistics
comparing model predictions with Stockfish evaluations.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import engine
import chess

def main():
    print("=" * 60)
    print("Stockfish Comparison Test")
    print("=" * 60)
    
    # Enable Stockfish comparison
    # You can specify the path to Stockfish if it's not in PATH
    print("\nEnabling Stockfish comparison...")
    engine.enable_stockfish_comparison(
        enabled=True,
        depth=10  # Stockfish search depth
    )
    
    # Reset stats to start fresh
    engine.reset_stockfish_comparison_stats()
    
    # Test positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian Game
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian Game
    ]
    
    print(f"\nEvaluating {len(test_positions)} positions with model...")
    print("(Stockfish will evaluate each position in the background for comparison)\n")
    
    # Evaluate positions with the model
    for i, fen in enumerate(test_positions, 1):
        print(f"Position {i}/{len(test_positions)}: {fen[:50]}...")
        
        # Use NNUE model if available, otherwise CNN
        if engine.load_nnue_model() is not None:
            model_score = engine.nnue_evaluate_model(fen)
            model_name = "NNUE"
        elif engine.load_cnn_model() is not None:
            model_score = engine.cnn_evaluate(fen)
            model_name = "CNN"
        else:
            model_score = engine.nnue_evaluate(fen)
            model_name = "Material"
        
        print(f"  {model_name} evaluation: {model_score} centipawns")
    
    # Print comparison statistics
    print("\n")
    engine.print_stockfish_comparison_stats()
    
    # Get detailed stats
    stats = engine.get_stockfish_comparison_stats()
    print(f"\nDetailed Statistics:")
    print(f"  Total comparisons: {stats['total_comparisons']}")
    if stats['total_comparisons'] > 0:
        print(f"  Mean Error: {stats['mean_error']:.2f} cp")
        print(f"  RMSE: {stats['rmse']:.2f} cp")
        print(f"  Max Error: {stats['max_error']:.2f} cp")
        print(f"  Correlation: {stats['correlation']:.4f}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    # Disable comparison to avoid slowing down normal engine usage
    engine.enable_stockfish_comparison(enabled=False)

if __name__ == "__main__":
    main()

