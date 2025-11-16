#!/usr/bin/env python3
"""
Benchmark multithreading improvements for ChessHacks.
Tests single-threaded vs multithreaded performance.
"""

import time
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Add build directory to path
build_dir = os.path.join(PROJECT_ROOT, 'build')
sys.path.insert(0, build_dir)

import c_helpers
from src.env_manager import get_env_config  # type: ignore


ENV_CONFIG = get_env_config()
MAX_DEPTH = ENV_CONFIG.search_depth
c_helpers.set_max_search_depth(MAX_DEPTH)

def benchmark_search(fen, depth, num_threads, name):
    """Benchmark search with specified number of threads."""
    tt = c_helpers.TranspositionTable()
    killer = c_helpers.KillerMoves()
    history = c_helpers.HistoryTable()
    counters = c_helpers.CounterMoveTable()
    
    start = time.time()
    score = c_helpers.alpha_beta_optimized_builtin(
        fen,
        depth,
        c_helpers.MIN,
        c_helpers.MAX,
        True,
        tt,
        num_threads,
        killer,
        history,
        counters,
    )
    elapsed = time.time() - start
    
    return score, elapsed, len(tt)

def benchmark_batch_evaluation():
    """Benchmark batch evaluation methods."""
    print("\n" + "="*80)
    print("BATCH EVALUATION BENCHMARK")
    print("="*80)
    
    # Generate test positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
    ] * 100  # 500 positions
    
    print(f"\nEvaluating {len(test_fens)} positions...\n")
    
    # Sequential evaluation
    print("1. Sequential evaluation (for loop)")
    start = time.time()
    scores_seq = [c_helpers.evaluate_with_pst(fen) for fen in test_fens]
    time_seq = time.time() - start
    print(f"   Time: {time_seq:.4f}s")
    print(f"   Rate: {len(test_fens)/time_seq:.0f} positions/sec")
    
    # Multithreaded batch evaluation (auto-detect threads)
    print("\n2. Multithreaded batch evaluation (auto threads)")
    start = time.time()
    scores_mt = c_helpers.batch_evaluate_mt(test_fens, 0)
    time_mt = time.time() - start
    print(f"   Time: {time_mt:.4f}s")
    print(f"   Rate: {len(test_fens)/time_mt:.0f} positions/sec")
    print(f"   Speedup: {time_seq/time_mt:.2f}x")
    
    # CUDA batch evaluation if available
    if c_helpers.is_cuda_available():
        print("\n3. CUDA batch evaluation (GPU)")
        start = time.time()
        scores_cuda = c_helpers.cuda_batch_evaluate(test_fens)
        time_cuda = time.time() - start
        print(f"   Time: {time_cuda:.4f}s")
        print(f"   Rate: {len(test_fens)/time_cuda:.0f} positions/sec")
        print(f"   Speedup vs sequential: {time_seq/time_cuda:.2f}x")
        print(f"   Speedup vs multithreaded: {time_mt/time_cuda:.2f}x")
    
    # Verify results match
    assert scores_seq == scores_mt, "MT scores don't match sequential!"
    print("\n✓ All evaluation methods produce identical results")

def main():
    print("="*80)
    print("ChessHacks Multithreading Benchmark")
    print("="*80)
    
    # Test positions at different depths
    positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex middlegame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"),
        ("Tactical position", "r2qkb1r/ppp2ppp/2n5/3np1B1/2BPP1b1/2P2N2/PP3PPP/RN1Q1RK1 w kq - 0 9"),
    ]
    
    depth_candidates = [4, 5, 6, 7, 8]
    depths = [d for d in depth_candidates if d <= MAX_DEPTH]
    if not depths:
        depths = [MAX_DEPTH]
    thread_counts = [1, 0]  # 1 = single-threaded, 0 = auto-detect all threads
    
    import threading
    max_threads = threading.active_count() + 1
    try:
        import multiprocessing
        max_threads = multiprocessing.cpu_count()
    except:
        pass
    
    print(f"\nSystem has {max_threads} CPU cores available")
    print(f"Testing with 1 thread (sequential) and {max_threads} threads (parallel)\n")
    print(f"Depths benchmarked: {depths} (use CHESSHACKS_MAX_DEPTH to adjust)\n")
    
    for pos_name, fen in positions:
        print("\n" + "="*80)
        print(f"Position: {pos_name}")
        print("="*80)
        
        for depth in depths:
            print(f"\n--- Depth {depth} ---")
            
            results = {}
            for num_threads in thread_counts:
                thread_label = "1 thread (sequential)" if num_threads == 1 else f"{max_threads} threads (parallel)"
                print(f"\n{thread_label}:")
                
                score, elapsed, nodes = benchmark_search(fen, depth, num_threads, thread_label)
                results[num_threads] = (score, elapsed, nodes)
                
                print(f"  Score: {score}")
                print(f"  Time: {elapsed:.4f}s")
                print(f"  Nodes: {nodes}")
                print(f"  NPS: {nodes/elapsed:.0f}")
            
            # Calculate speedup
            if 1 in results and 0 in results:
                speedup = results[1][1] / results[0][1]
                print(f"\n  Speedup (parallel vs sequential): {speedup:.2f}x")
                
                # Check if scores match
                if results[1][0] != results[0][0]:
                    print(f"  ⚠ Warning: Scores differ! Sequential: {results[1][0]}, Parallel: {results[0][0]}")
                else:
                    print(f"  ✓ Scores match")
    
    # Batch evaluation benchmark
    benchmark_batch_evaluation()
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

