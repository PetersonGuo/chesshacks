#include "cuda/cuda_eval.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace {

constexpr int BOARD_STRIDE = 64;

void WriteBoardToBuffer(const bitboard::BitboardState &board, int8_t *dest) {
  for (int sq = 0; sq < BOARD_STRIDE; ++sq) {
    dest[sq] = static_cast<int8_t>(board.get_piece_at(sq));
  }
}

std::vector<int8_t>
BoardsToPieceBuffer(const std::vector<bitboard::BitboardState> &boards) {
  std::vector<int8_t> packed;
  packed.resize(static_cast<size_t>(boards.size()) * BOARD_STRIDE);
  for (size_t i = 0; i < boards.size(); ++i) {
    WriteBoardToBuffer(boards[i], packed.data() + i * BOARD_STRIDE);
  }
  return packed;
}

} // namespace

// Material values
__constant__ int d_piece_values[6] = {100, 320, 330, 500, 900, 20000};

// ============================================================================
// CUDA KERNEL: Batch Position Evaluation
// ============================================================================

/**
 * Optimized CUDA kernel for batch evaluation of chess positions
 * Uses shared memory for better performance and coalesced memory access
 * Each thread evaluates one position
 */
__global__ void
batch_evaluate_kernel(const int8_t *boards, // Flattened board components
                      int *scores,          // Output scores
                      int num_positions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_positions)
    return;

  // Use shared memory for piece values to reduce global memory access
  __shared__ int s_piece_values[6];

  // Load piece values into shared memory (first 6 threads)
  if (threadIdx.x < 6) {
    s_piece_values[threadIdx.x] = d_piece_values[threadIdx.x];
  }

  __syncthreads();

  const int8_t *board = boards + idx * BOARD_STRIDE;

  int score = 0;

#pragma unroll 4
  for (int square = 0; square < BOARD_STRIDE; ++square) {
    int piece = static_cast<int>(board[square]);
    if (piece == 0)
      continue;

    bool is_white = piece > 0;
    int index = is_white ? piece : -piece;
    if (index >= 1 && index <= 6) {
      int piece_value = s_piece_values[index - 1];
      score += is_white ? piece_value : -piece_value;
    }
  }

  scores[idx] = score;
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

/**
 * Batch evaluate positions on GPU
 */
bool cuda_batch_evaluate(const std::vector<bitboard::BitboardState> &boards,
                         std::vector<int> &scores) {
  if (boards.empty())
    return false;

  int num_positions = static_cast<int>(boards.size());
  scores.resize(num_positions);

  std::vector<int8_t> flat_boards = BoardsToPieceBuffer(boards);

  // Allocate device memory
  int8_t *d_positions = nullptr;
  int *d_scores = nullptr;

  cudaError_t err;
  err = cudaMalloc(&d_positions,
                   flat_boards.size() * static_cast<int>(sizeof(int8_t)));
  if (err != cudaSuccess)
    return false;

  err = cudaMalloc(&d_scores, num_positions * sizeof(int));
  if (err != cudaSuccess) {
    cudaFree(d_positions);
    return false;
  }

  // Copy board components to device
  err = cudaMemcpy(d_positions, flat_boards.data(),
                   flat_boards.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_positions);
    cudaFree(d_scores);
    return false;
  }

  // Launch kernel with optimized parameters
  // Use 256 threads per block for optimal occupancy with shared memory
  // This allows efficient loading of piece-square tables
  int threads_per_block = 256;
  int num_blocks = (num_positions + threads_per_block - 1) / threads_per_block;

  // Calculate shared memory size (for documentation/future use)
  // 6 ints (piece values) = 24 bytes, well within shared memory limits

  batch_evaluate_kernel<<<num_blocks, threads_per_block>>>(
      d_positions, d_scores, num_positions);

  // Check for kernel errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_positions);
    cudaFree(d_scores);
    return false;
  }

  // Wait for completion
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_positions);
    cudaFree(d_scores);
    return false;
  }

  // Copy results back
  err = cudaMemcpy(scores.data(), d_scores, num_positions * sizeof(int),
                   cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_scores);

  return (err == cudaSuccess);
}

/**
 * Single position evaluation using GPU (for testing)
 */
int cuda_evaluate_position(const bitboard::BitboardState &board) {
  std::vector<bitboard::BitboardState> boards = {board};
  std::vector<int> scores;

  if (cuda_batch_evaluate(boards, scores)) {
    return scores[0];
  }

  return 0; // Error fallback
}

// ============================================================================
// CUDA KERNEL: Batch Move Scoring (MVV-LVA)
// ============================================================================

/**
 * Device function to get piece value on GPU
 */
__device__ int d_get_piece_value(int8_t piece) {
  int abs_piece = piece >= 0 ? piece : -piece;
  switch (abs_piece) {
  case 1:
    return 100;
  case 2:
    return 320;
  case 3:
    return 330;
  case 4:
    return 500;
  case 5:
    return 900;
  case 6:
    return 20000;
  default:
    return 0;
  }
}

/**
 * CUDA kernel for batch MVV-LVA move scoring
 * Scores capture moves based on victim value and attacker value
 */
__global__ void batch_mvv_lva_kernel(
    const int8_t *boards,    // Flattened board positions (64 entries each)
    const int *from_squares, // Source squares
    const int *to_squares,   // Destination squares
    int *scores,             // Output MVV-LVA scores
    int num_moves) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_moves)
    return;

  const int8_t *board = boards + idx * BOARD_STRIDE;
  int from = from_squares[idx];
  int to = to_squares[idx];

  // Get attacker and victim pieces
  int8_t attacker = board[from];
  int8_t victim = board[to];

  // Calculate MVV-LVA score: victim_value * 10 - attacker_value
  int victim_value = d_get_piece_value(victim);
  int attacker_value = d_get_piece_value(attacker);

  scores[idx] = victim_value * 10 - attacker_value;
}

// ============================================================================
// CUDA KERNEL: Parallel Move Ordering
// ============================================================================

/**
 * Bitonic sort kernel for move ordering
 * Sorts moves by their scores in descending order
 */
__global__ void bitonic_sort_moves_kernel(int *scores, int *move_indices, int n,
                                          int k, int j) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int ixj = idx ^ j;

  if (ixj > idx && idx < n && ixj < n) {
    bool ascending = ((idx & k) == 0);

    if ((scores[idx] < scores[ixj]) == ascending) {
      // Swap scores
      int temp_score = scores[idx];
      scores[idx] = scores[ixj];
      scores[ixj] = temp_score;

      // Swap indices
      int temp_idx = move_indices[idx];
      move_indices[idx] = move_indices[ixj];
      move_indices[ixj] = temp_idx;
    }
  }
}

// ============================================================================
// CUDA KERNEL: Parallel Piece Counting
// ============================================================================

/**
 * Optimized CUDA kernel for counting pieces of each type in multiple positions
 * Useful for material evaluation and endgame detection
 * Uses registers for local counting before writing to global memory
 */
__global__ void batch_count_pieces_kernel(
    const int8_t *boards, // Flattened board components
    int *piece_counts,    // Output: [num_pos][12] (6 white, 6 black)
    int num_positions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_positions)
    return;

  const int8_t *board = boards + idx * BOARD_STRIDE;
  int counts[12] = {0};

#pragma unroll 4
  for (int sq = 0; sq < BOARD_STRIDE; ++sq) {
    int piece = static_cast<int>(board[sq]);
    if (piece > 0) {
      int piece_idx = piece - 1;
      if (piece_idx >= 0 && piece_idx < 6) {
        counts[piece_idx]++;
      }
    } else if (piece < 0) {
      int piece_idx = (-piece) - 1;
      if (piece_idx >= 0 && piece_idx < 6) {
        counts[6 + piece_idx]++;
      }
    }
  }

  int *output = piece_counts + idx * 12;
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    output[i] = counts[i];
  }
}

// ============================================================================
// CUDA KERNEL: Batch Static Exchange Evaluation (SEE)
// ============================================================================

/**
 * Simplified SEE evaluation on GPU
 * Evaluates if a capture is good or bad
 */
__global__ void batch_see_kernel(const char *boards,      // Board positions
                                 const int *from_squares, // Capture source
                                 const int *to_squares,   // Capture destination
                                 int *see_scores,         // Output SEE scores
                                 int num_moves) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_moves)
    return;

  const char *board = boards + (idx / 64) * 64;
  int from = from_squares[idx];
  int to = to_squares[idx];

  char attacker = board[from];
  char victim = board[to];

  // Simplified SEE: just compare piece values
  // Full SEE would require attack detection which is complex on GPU
  int gain = d_get_piece_value(victim);
  int risk = d_get_piece_value(attacker);

  // Positive if we gain material, negative if we lose
  see_scores[idx] = gain - risk;
}

// ============================================================================
// CUDA KERNEL: Parallel Position Hashing
// ============================================================================

/**
 * Simple hash function for chess positions (Zobrist-like)
 * Used for transposition table lookups
 */
__global__ void batch_hash_positions_kernel(const int8_t *boards,
                                            unsigned long long *hashes,
                                            int num_positions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_positions)
    return;

  const int8_t *board = boards + idx * BOARD_STRIDE;
  unsigned long long hash = 1469598103934665603ull;

#pragma unroll 4
  for (int sq = 0; sq < BOARD_STRIDE; ++sq) {
    hash ^= static_cast<unsigned long long>(board[sq] + 16);
    hash *= 1099511628211ull;
  }

  hashes[idx] = hash;
}

// ============================================================================
// HOST FUNCTIONS: Batch Operations
// ============================================================================

/**
 * Batch MVV-LVA scoring from host
 */
bool cuda_batch_mvv_lva(const std::vector<bitboard::BitboardState> &boards,
                        const std::vector<int> &from_squares,
                        const std::vector<int> &to_squares,
                        std::vector<int> &scores) {
  int num_moves = from_squares.size();
  if (num_moves == 0 || boards.empty())
    return false;

  std::vector<int8_t> flat_boards;
  if (boards.size() == 1 && num_moves > 1) {
    flat_boards.resize(static_cast<size_t>(num_moves) * BOARD_STRIDE);
    for (int i = 0; i < num_moves; ++i) {
      WriteBoardToBuffer(boards[0], flat_boards.data() + i * BOARD_STRIDE);
    }
  } else if (boards.size() == static_cast<size_t>(num_moves)) {
    flat_boards = BoardsToPieceBuffer(boards);
  } else {
    return false;
  }

  int8_t *d_boards = nullptr;
  int *d_from = nullptr;
  int *d_to = nullptr;
  int *d_scores = nullptr;

  cudaError_t err = cudaMalloc(&d_boards, flat_boards.size() *
                                              static_cast<int>(sizeof(int8_t)));
  if (err != cudaSuccess)
    return false;

  err = cudaMalloc(&d_from, num_moves * sizeof(int));
  if (err != cudaSuccess)
    goto cleanup;
  err = cudaMalloc(&d_to, num_moves * sizeof(int));
  if (err != cudaSuccess)
    goto cleanup;
  err = cudaMalloc(&d_scores, num_moves * sizeof(int));
  if (err != cudaSuccess)
    goto cleanup;

  err = cudaMemcpy(d_boards, flat_boards.data(),
                   flat_boards.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    goto cleanup;
  err = cudaMemcpy(d_from, from_squares.data(), num_moves * sizeof(int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    goto cleanup;
  err = cudaMemcpy(d_to, to_squares.data(), num_moves * sizeof(int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    goto cleanup;

  {
    int threads_per_block = 256;
    int num_blocks = (num_moves + threads_per_block - 1) / threads_per_block;
    batch_mvv_lva_kernel<<<num_blocks, threads_per_block>>>(
        d_boards, d_from, d_to, d_scores, num_moves);
    err = cudaGetLastError();
    if (err != cudaSuccess)
      goto cleanup;
  }

  scores.resize(num_moves);
  err = cudaMemcpy(scores.data(), d_scores, num_moves * sizeof(int),
                   cudaMemcpyDeviceToHost);

cleanup:
  if (d_boards)
    cudaFree(d_boards);
  if (d_from)
    cudaFree(d_from);
  if (d_to)
    cudaFree(d_to);
  if (d_scores)
    cudaFree(d_scores);

  return err == cudaSuccess;
}

/**
 * Batch piece counting from host
 */
bool cuda_batch_count_pieces(const std::vector<bitboard::BitboardState> &boards,
                             std::vector<std::vector<int>> &piece_counts) {
  int num_positions = boards.size();
  if (num_positions == 0)
    return false;

  std::vector<int8_t> flat_boards = BoardsToPieceBuffer(boards);

  // Allocate device memory
  int8_t *d_positions;
  int *d_counts;
  cudaMalloc(&d_positions,
             flat_boards.size() * static_cast<int>(sizeof(int8_t)));
  cudaMalloc(&d_counts, num_positions * 12 * sizeof(int));

  // Copy to device
  cudaMemcpy(d_positions, flat_boards.data(),
             flat_boards.size() * sizeof(int8_t), cudaMemcpyHostToDevice);

  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (num_positions + threads_per_block - 1) / threads_per_block;

  batch_count_pieces_kernel<<<num_blocks, threads_per_block>>>(
      d_positions, d_counts, num_positions);

  cudaDeviceSynchronize();

  // Copy results back
  piece_counts.resize(num_positions);
  std::vector<int> all_counts(num_positions * 12);
  cudaMemcpy(all_counts.data(), d_counts, num_positions * 12 * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_positions; i++) {
    piece_counts[i].assign(all_counts.begin() + i * 12,
                           all_counts.begin() + (i + 1) * 12);
  }

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_counts);

  return true;
}

/**
 * Batch position hashing from host
 */
bool cuda_batch_hash_positions(
    const std::vector<bitboard::BitboardState> &boards,
    std::vector<unsigned long long> &hashes) {
  int num_positions = boards.size();
  if (num_positions == 0)
    return false;

  std::vector<int8_t> flat_boards = BoardsToPieceBuffer(boards);

  // Allocate device memory
  int8_t *d_positions;
  unsigned long long *d_hashes;
  cudaMalloc(&d_positions,
             flat_boards.size() * static_cast<int>(sizeof(int8_t)));
  cudaMalloc(&d_hashes, num_positions * sizeof(unsigned long long));

  // Copy to device
  cudaMemcpy(d_positions, flat_boards.data(),
             flat_boards.size() * sizeof(int8_t), cudaMemcpyHostToDevice);

  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (num_positions + threads_per_block - 1) / threads_per_block;

  batch_hash_positions_kernel<<<num_blocks, threads_per_block>>>(
      d_positions, d_hashes, num_positions);

  cudaDeviceSynchronize();

  // Copy results back
  hashes.resize(num_positions);
  cudaMemcpy(hashes.data(), d_hashes,
             num_positions * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_hashes);

  return true;
}

// Python-friendly wrappers
std::vector<int>
cuda_batch_evaluate_py(const std::vector<bitboard::BitboardState> &boards) {
  std::vector<int> scores;
  cuda_batch_evaluate(boards, scores);
  return scores;
}

std::vector<std::vector<int>>
cuda_batch_count_pieces_py(const std::vector<bitboard::BitboardState> &boards) {
  std::vector<std::vector<int>> piece_counts;
  cuda_batch_count_pieces(boards, piece_counts);
  return piece_counts;
}

std::vector<unsigned long long> cuda_batch_hash_positions_py(
    const std::vector<bitboard::BitboardState> &boards) {
  std::vector<unsigned long long> hashes;
  cuda_batch_hash_positions(boards, hashes);
  return hashes;
}
