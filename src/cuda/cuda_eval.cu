#include "../bitboard/bitboard_state.h"
#include "../evaluation.h"
#include "cuda_eval.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>

// ============================================================================
// CUDA PIECE-SQUARE TABLES (Device Constants)
// ============================================================================

// Device constant memory for piece-square tables (faster access)
__constant__ int d_pawn_table[64];
__constant__ int d_knight_table[64];
__constant__ int d_bishop_table[64];
__constant__ int d_rook_table[64];
__constant__ int d_queen_table[64];
__constant__ int d_king_table[64];

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
batch_evaluate_kernel(const char *positions, // Flattened FEN strings
                      int *scores,           // Output scores
                      int num_positions, int max_fen_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_positions)
    return;

  // Use shared memory for piece-square tables to reduce global memory access
  __shared__ int s_piece_values[6];
  __shared__ int s_pawn_table[64];
  __shared__ int s_knight_table[64];
  __shared__ int s_bishop_table[64];
  __shared__ int s_rook_table[64];
  __shared__ int s_queen_table[64];
  __shared__ int s_king_table[64];

  // Load piece values into shared memory (first 6 threads)
  if (threadIdx.x < 6) {
    s_piece_values[threadIdx.x] = d_piece_values[threadIdx.x];
  }

  // Load piece-square tables into shared memory (all threads participate)
  int tid = threadIdx.x;
  if (tid < 64) {
    s_pawn_table[tid] = d_pawn_table[tid];
    s_knight_table[tid] = d_knight_table[tid];
    s_bishop_table[tid] = d_bishop_table[tid];
    s_rook_table[tid] = d_rook_table[tid];
    s_queen_table[tid] = d_queen_table[tid];
    s_king_table[tid] = d_king_table[tid];
  }

  __syncthreads();

  // Get this thread's FEN string - coalesced memory access
  const char *fen = positions + idx * max_fen_length;

  int score = 0;
  int rank = 7; // Start from rank 8 (index 7)
  int file = 0;
  int i = 0;

  // Parse FEN and evaluate
  #pragma unroll 4
  while (fen[i] != '\0' && fen[i] != ' ' && i < max_fen_length) {
    char c = fen[i];

    if (c == '/') {
      rank--;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += (c - '0');
    } else {
      // It's a piece
      int square = rank * 8 + file;
      int piece_value = 0;
      int pst_value = 0;
      bool is_white = (c >= 'A' && c <= 'Z');

      // Calculate square index (flip for black)
      int pst_square = is_white ? square : (63 - square);

      // Get piece type and values - use shared memory
      char piece = is_white ? c : (c - 32); // Convert to uppercase

      switch (piece) {
      case 'P':
        piece_value = s_piece_values[0];
        pst_value = s_pawn_table[pst_square];
        break;
      case 'N':
        piece_value = s_piece_values[1];
        pst_value = s_knight_table[pst_square];
        break;
      case 'B':
        piece_value = s_piece_values[2];
        pst_value = s_bishop_table[pst_square];
        break;
      case 'R':
        piece_value = s_piece_values[3];
        pst_value = s_rook_table[pst_square];
        break;
      case 'Q':
        piece_value = s_piece_values[4];
        pst_value = s_queen_table[pst_square];
        break;
      case 'K':
        piece_value = s_piece_values[5];
        pst_value = s_king_table[pst_square];
        break;
      }

      // Add to score (positive for white, negative for black)
      int total = piece_value + pst_value;
      score += is_white ? total : -total;

      file++;
    }
    i++;
  }

  scores[idx] = score;
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

/**
 * Initialize CUDA device and copy piece-square tables to device constant memory
 */
bool cuda_init_tables(const int *pawn_table, const int *knight_table,
                      const int *bishop_table, const int *rook_table,
                      const int *queen_table, const int *king_table) {
  cudaError_t err;

  err = cudaMemcpyToSymbol(d_pawn_table, pawn_table, 64 * sizeof(int));
  if (err != cudaSuccess)
    return false;

  err = cudaMemcpyToSymbol(d_knight_table, knight_table, 64 * sizeof(int));
  if (err != cudaSuccess)
    return false;

  err = cudaMemcpyToSymbol(d_bishop_table, bishop_table, 64 * sizeof(int));
  if (err != cudaSuccess)
    return false;

  err = cudaMemcpyToSymbol(d_rook_table, rook_table, 64 * sizeof(int));
  if (err != cudaSuccess)
    return false;

  err = cudaMemcpyToSymbol(d_queen_table, queen_table, 64 * sizeof(int));
  if (err != cudaSuccess)
    return false;

  err = cudaMemcpyToSymbol(d_king_table, king_table, 64 * sizeof(int));
  if (err != cudaSuccess)
    return false;

  return true;
}

/**
 * Batch evaluate positions on GPU
 */
bool cuda_batch_evaluate(const std::vector<std::string> &fens,
                         std::vector<int> &scores) {
  if (fens.empty())
    return false;

  int num_positions = fens.size();
  scores.resize(num_positions);

  // Find max FEN length
  int max_fen_length = 0;
  for (const auto &fen : fens) {
    max_fen_length = std::max(max_fen_length, (int)fen.length() + 1);
  }

  // Flatten FEN strings into contiguous buffer
  std::vector<char> flat_fens(num_positions * max_fen_length, '\0');
  for (int i = 0; i < num_positions; i++) {
    strncpy(&flat_fens[i * max_fen_length], fens[i].c_str(),
            max_fen_length - 1);
  }

  // Allocate device memory
  char *d_positions = nullptr;
  int *d_scores = nullptr;

  cudaError_t err;
  err = cudaMalloc(&d_positions, flat_fens.size());
  if (err != cudaSuccess)
    return false;

  err = cudaMalloc(&d_scores, num_positions * sizeof(int));
  if (err != cudaSuccess) {
    cudaFree(d_positions);
    return false;
  }

  // Copy FENs to device
  err = cudaMemcpy(d_positions, flat_fens.data(), flat_fens.size(),
                   cudaMemcpyHostToDevice);
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
  // 6 ints (piece values) + 6*64 ints (PST tables) = 390 ints = 1560 bytes
  // This fits well within typical 48KB shared memory per block

  batch_evaluate_kernel<<<num_blocks, threads_per_block>>>(
      d_positions, d_scores, num_positions, max_fen_length);

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
int cuda_evaluate_position(const std::string &fen) {
  std::vector<std::string> fens = {fen};
  std::vector<int> scores;

  if (cuda_batch_evaluate(fens, scores)) {
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
__device__ int d_get_piece_value(char piece) {
  char p = (piece >= 'a' && piece <= 'z') ? (piece - 32) : piece;

  switch (p) {
  case 'P':
    return 100;
  case 'N':
    return 320;
  case 'B':
    return 330;
  case 'R':
    return 500;
  case 'Q':
    return 900;
  case 'K':
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
    const char *boards,      // Flattened board positions (64 bytes each)
    const int *from_squares, // Source squares
    const int *to_squares,   // Destination squares
    int *scores,             // Output MVV-LVA scores
    int num_moves) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_moves)
    return;

  const char *board =
      boards + (idx / 64) * 64; // Each move might be on different board
  int from = from_squares[idx];
  int to = to_squares[idx];

  // Get attacker and victim pieces
  char attacker = board[from];
  char victim = board[to];

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
    const char *positions, // Flattened FEN strings
    int *piece_counts,     // Output: [num_pos][12] (6 white, 6 black)
    int num_positions, int max_fen_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_positions)
    return;

  const char *fen = positions + idx * max_fen_length;

  // Use registers for local counting (much faster than global memory)
  int counts[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Parse FEN and count pieces
  int i = 0;
  #pragma unroll 4
  while (fen[i] != '\0' && fen[i] != ' ' && i < max_fen_length) {
    char c = fen[i];

    if (c >= 'A' && c <= 'Z') {
      // White pieces (indices 0-5)
      switch (c) {
      case 'P':
        counts[0]++;
        break;
      case 'N':
        counts[1]++;
        break;
      case 'B':
        counts[2]++;
        break;
      case 'R':
        counts[3]++;
        break;
      case 'Q':
        counts[4]++;
        break;
      case 'K':
        counts[5]++;
        break;
      }
    } else if (c >= 'a' && c <= 'z') {
      // Black pieces (indices 6-11)
      switch (c) {
      case 'p':
        counts[6]++;
        break;
      case 'n':
        counts[7]++;
        break;
      case 'b':
        counts[8]++;
        break;
      case 'r':
        counts[9]++;
        break;
      case 'q':
        counts[10]++;
        break;
      case 'k':
        counts[11]++;
        break;
      }
    }

    i++;
  }

  // Write results to global memory in one coalesced transaction
  int *output = piece_counts + idx * 12;
  #pragma unroll
  for (int i = 0; i < 12; i++) {
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
__global__ void batch_hash_positions_kernel(const char *positions,
                                            unsigned long long *hashes,
                                            int num_positions,
                                            int max_fen_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_positions)
    return;

  const char *fen = positions + idx * max_fen_length;
  unsigned long long hash = 0;

  // Simple polynomial rolling hash
  int i = 0;
  while (fen[i] != '\0' && fen[i] != ' ' && i < max_fen_length) {
    hash = hash * 31 + fen[i];
    i++;
  }

  hashes[idx] = hash;
}

// ============================================================================
// HOST FUNCTIONS: Batch Operations
// ============================================================================

/**
 * Batch MVV-LVA scoring from host
 */
bool cuda_batch_mvv_lva(const std::vector<std::string> &boards,
                        const std::vector<int> &from_squares,
                        const std::vector<int> &to_squares,
                        std::vector<int> &scores) {
  int num_moves = from_squares.size();
  if (num_moves == 0 || boards.empty())
    return false;

  // Prepare board data (convert FEN to 64-byte board representation)
  // For now, we'll just pass FEN strings
  // TODO: Convert to actual board arrays for efficiency

  // Allocate device memory
  int *d_from, *d_to, *d_scores;
  cudaMalloc(&d_from, num_moves * sizeof(int));
  cudaMalloc(&d_to, num_moves * sizeof(int));
  cudaMalloc(&d_scores, num_moves * sizeof(int));

  // Copy data to device
  cudaMemcpy(d_from, from_squares.data(), num_moves * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_to, to_squares.data(), num_moves * sizeof(int),
             cudaMemcpyHostToDevice);

  // Launch kernel (simplified - needs proper board data)
  // int threads_per_block = 256;
  // int num_blocks = (num_moves + threads_per_block - 1) / threads_per_block;

  // Note: This is a placeholder - we need proper board representation
  // batch_mvv_lva_kernel<<<num_blocks, threads_per_block>>>(nullptr, d_from,
  // d_to, d_scores, num_moves);

  // Copy results back
  scores.resize(num_moves);
  cudaMemcpy(scores.data(), d_scores, num_moves * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_from);
  cudaFree(d_to);
  cudaFree(d_scores);

  return true;
}

/**
 * Batch piece counting from host
 */
bool cuda_batch_count_pieces(const std::vector<std::string> &fens,
                             std::vector<std::vector<int>> &piece_counts) {
  int num_positions = fens.size();
  if (num_positions == 0)
    return false;

  // Find maximum FEN length
  size_t max_len = 0;
  for (const auto &fen : fens) {
    if (fen.size() > max_len)
      max_len = fen.size();
  }
  max_len++; // For null terminator

  // Prepare padded FEN strings
  std::vector<char> padded_fens(num_positions * max_len, '\0');
  for (int i = 0; i < num_positions; i++) {
    std::memcpy(padded_fens.data() + i * max_len, fens[i].c_str(),
                fens[i].size());
  }

  // Allocate device memory
  char *d_positions;
  int *d_counts;
  cudaMalloc(&d_positions, num_positions * max_len);
  cudaMalloc(&d_counts, num_positions * 12 * sizeof(int));

  // Copy to device
  cudaMemcpy(d_positions, padded_fens.data(), num_positions * max_len,
             cudaMemcpyHostToDevice);

  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (num_positions + threads_per_block - 1) / threads_per_block;

  batch_count_pieces_kernel<<<num_blocks, threads_per_block>>>(
      d_positions, d_counts, num_positions, max_len);

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
bool cuda_batch_hash_positions(const std::vector<std::string> &fens,
                               std::vector<unsigned long long> &hashes) {
  int num_positions = fens.size();
  if (num_positions == 0)
    return false;

  // Find maximum FEN length
  size_t max_len = 0;
  for (const auto &fen : fens) {
    if (fen.size() > max_len)
      max_len = fen.size();
  }
  max_len++;

  // Prepare padded FEN strings
  std::vector<char> padded_fens(num_positions * max_len, '\0');
  for (int i = 0; i < num_positions; i++) {
    std::memcpy(padded_fens.data() + i * max_len, fens[i].c_str(),
                fens[i].size());
  }

  // Allocate device memory
  char *d_positions;
  unsigned long long *d_hashes;
  cudaMalloc(&d_positions, num_positions * max_len);
  cudaMalloc(&d_hashes, num_positions * sizeof(unsigned long long));

  // Copy to device
  cudaMemcpy(d_positions, padded_fens.data(), num_positions * max_len,
             cudaMemcpyHostToDevice);

  // Launch kernel
  int threads_per_block = 256;
  int num_blocks = (num_positions + threads_per_block - 1) / threads_per_block;

  batch_hash_positions_kernel<<<num_blocks, threads_per_block>>>(
      d_positions, d_hashes, num_positions, max_len);

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
std::vector<int> cuda_batch_evaluate_py(const std::vector<std::string> &fens) {
  std::vector<int> scores;
  cuda_batch_evaluate(fens, scores);
  return scores;
}

std::vector<std::vector<int>>
cuda_batch_count_pieces_py(const std::vector<std::string> &fens) {
  std::vector<std::vector<int>> piece_counts;
  cuda_batch_count_pieces(fens, piece_counts);
  return piece_counts;
}

std::vector<unsigned long long>
cuda_batch_hash_positions_py(const std::vector<std::string> &fens) {
  std::vector<unsigned long long> hashes;
  cuda_batch_hash_positions(fens, hashes);
  return hashes;
}
