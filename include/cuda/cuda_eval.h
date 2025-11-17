#ifndef CUDA_EVAL_H
#define CUDA_EVAL_H

#include "../bitboard/bitboard_state.h"
#include <vector>

/**
 * Batch evaluate multiple chess positions on GPU
 * Much more efficient than evaluating one at a time
 *
 * @param boards Vector of BitboardState positions to evaluate
 * @param scores Output vector of evaluation scores (resized automatically)
 * @return true if successful, false otherwise
 */
bool cuda_batch_evaluate(const std::vector<bitboard::BitboardState> &boards,
                         std::vector<int> &scores);

/**
 * Evaluate a single position using GPU
 * Less efficient than batch evaluation but easier to use
 *
 * @param board BitboardState to evaluate
 * @return Evaluation score
 */
int cuda_evaluate_position(const bitboard::BitboardState &board);

/**
 * Batch MVV-LVA move scoring on GPU
 * Scores capture moves based on Most Valuable Victim - Least Valuable Attacker
 *
 * @param boards Vector of BitboardState positions (either 1 or per-move)
 * @param from_squares Vector of source squares (0-63)
 * @param to_squares Vector of destination squares (0-63)
 * @param scores Output vector of MVV-LVA scores
 * @return true if successful, false otherwise
 */
bool cuda_batch_mvv_lva(const std::vector<bitboard::BitboardState> &boards,
                        const std::vector<int> &from_squares,
                        const std::vector<int> &to_squares,
                        std::vector<int> &scores);

/**
 * Batch piece counting on GPU
 * Counts pieces of each type in multiple positions
 * Useful for material evaluation and endgame detection
 *
 * @param boards Vector of BitboardState positions
 * @param piece_counts Output: vector of [12] int arrays (6 white, 6 black
 * pieces)
 * @return true if successful, false otherwise
 */
bool cuda_batch_count_pieces(const std::vector<bitboard::BitboardState> &boards,
                             std::vector<std::vector<int>> &piece_counts);

/**
 * Batch position hashing on GPU
 * Generates hash values for transposition table lookups
 *
 * @param boards Vector of BitboardState positions
 * @param hashes Output vector of hash values
 * @return true if successful, false otherwise
 */
bool cuda_batch_hash_positions(
    const std::vector<bitboard::BitboardState> &boards,
    std::vector<unsigned long long> &hashes);

// Python-friendly wrappers that return values instead of using output
// parameters
std::vector<int>
cuda_batch_evaluate_py(const std::vector<bitboard::BitboardState> &boards);
std::vector<std::vector<int>>
cuda_batch_count_pieces_py(const std::vector<bitboard::BitboardState> &boards);
std::vector<unsigned long long> cuda_batch_hash_positions_py(
    const std::vector<bitboard::BitboardState> &boards);

#endif // CUDA_EVAL_H
