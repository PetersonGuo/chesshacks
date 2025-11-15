from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import sys
import os

# Add build directory to path so c_helpers can be imported
build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
if build_path not in sys.path:
    sys.path.insert(0, build_path)

import c_helpers

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass


def main():
    def nnue_evaluate(fen: str) -> int:
        # Placeholder: return a simple evaluation
        return 0

    # Example usage with FEN string and NNUE callback
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    result = c_helpers.alpha_beta(starting_fen, 10, c_helpers.MIN, c_helpers.MAX, True, nnue_evaluate)
    print(f"Alpha-beta result: {result}")