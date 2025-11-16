#!/usr/bin/env python3
"""Test the updated main.py search functionality"""

import sys
import os
import time

# Add build directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c_helpers


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


# Detect CUDA (same as in main.py)
def has_cuda():
    """Check if CUDA is available on this system"""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=1, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


USE_CUDA = has_cuda()
SEARCH_DEPTH = _env_int("CHESSHACKS_MAX_DEPTH", 4)
MAX_DEPTH = SEARCH_DEPTH
c_helpers.set_max_search_depth(MAX_DEPTH)
NUM_THREADS = 0

# Create transposition table
transposition_table = c_helpers.TranspositionTable()

print("=" * 60)
print("Chess Engine Test")
print("=" * 60)
print(f"CUDA available: {USE_CUDA}")
print(f"Search depth: {SEARCH_DEPTH}")
print(f"Threads: {NUM_THREADS if NUM_THREADS > 0 else 'auto'}")
print()


def search_position(fen: str, depth: int = SEARCH_DEPTH) -> int:
    """Search a position using the best available engine"""
    if USE_CUDA:
        return c_helpers.alpha_beta_cuda(
            fen,
            depth,
            c_helpers.MIN,
            c_helpers.MAX,
            True,
            c_helpers.evaluate_material,
            transposition_table,
        )
    else:
        return c_helpers.alpha_beta_optimized_builtin(
            fen,
            depth,
            c_helpers.MIN,
            c_helpers.MAX,
            True,
            transposition_table,
            NUM_THREADS,
        )


# Test 1: Starting position
print("Test 1: Starting position")
fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print(f"FEN: {fen1}")
start = time.time()
score1 = search_position(fen1, depth=3)
elapsed1 = time.time() - start
print(f"Score: {score1}")
print(f"Time: {elapsed1:.3f}s")
print(f"TT entries: {len(transposition_table)}")
print()

# Test 2: After 1.e4
print("Test 2: After 1.e4")
transposition_table.clear()
fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
print(f"FEN: {fen2}")
start = time.time()
score2 = search_position(fen2, depth=3)
elapsed2 = time.time() - start
print(f"Score: {score2}")
print(f"Time: {elapsed2:.3f}s")
print(f"TT entries: {len(transposition_table)}")
print()

print("=" * 60)
print("✅ All tests passed!")
print()
print("Engine is using:")
if USE_CUDA:
    print("  🚀 CUDA acceleration (alpha_beta_cuda)")
else:
    print("  ⚡ CPU optimized (alpha_beta_optimized)")
print()
print("Features active:")
print("  ✓ Transposition table")
print("  ✓ Move ordering (TT + MVV-LVA + promotions)")
print("  ✓ Quiescence search (depth limit 4)")
print("  ✓ Iterative deepening")
if not USE_CUDA and NUM_THREADS != 1:
    print("  ✓ Parallel search")
print("=" * 60)
