#ifndef CUDA_EVAL_H
#define CUDA_EVAL_H

#include <string>
#include <vector>

/**
 * Initialize CUDA device and copy piece-square tables to device constant memory
 * Must be called once before using CUDA evaluation functions
 *
 * @param pawn_table Array of 64 ints for pawn PST
 * @param knight_table Array of 64 ints for knight PST
 * @param bishop_table Array of 64 ints for bishop PST
 * @param rook_table Array of 64 ints for rook PST
 * @param queen_table Array of 64 ints for queen PST
 * @param king_table Array of 64 ints for king PST
 * @return true if successful, false otherwise
 */
bool cuda_init_tables(const int *pawn_table, const int *knight_table,
                      const int *bishop_table, const int *rook_table,
                      const int *queen_table, const int *king_table);

/**
 * Batch evaluate multiple chess positions on GPU
 * Much more efficient than evaluating one at a time
 *
 * @param fens Vector of FEN strings to evaluate
 * @param scores Output vector of evaluation scores (resized automatically)
 * @return true if successful, false otherwise
 */
bool cuda_batch_evaluate(const std::vector<std::string> &fens,
                         std::vector<int> &scores);

/**
 * Evaluate a single position using GPU
 * Less efficient than batch evaluation but easier to use
 *
 * @param fen FEN string to evaluate
 * @return Evaluation score
 */
int cuda_evaluate_position(const std::string &fen);

/**
 * Batch MVV-LVA move scoring on GPU
 * Scores capture moves based on Most Valuable Victim - Least Valuable Attacker
 *
 * @param boards Vector of board position strings (FEN)
 * @param from_squares Vector of source squares (0-63)
 * @param to_squares Vector of destination squares (0-63)
 * @param scores Output vector of MVV-LVA scores
 * @return true if successful, false otherwise
 */
bool cuda_batch_mvv_lva(const std::vector<std::string> &boards,
                        const std::vector<int> &from_squares,
                        const std::vector<int> &to_squares,
                        std::vector<int> &scores);

/**
 * Batch piece counting on GPU
 * Counts pieces of each type in multiple positions
 * Useful for material evaluation and endgame detection
 *
 * @param fens Vector of FEN strings
 * @param piece_counts Output: vector of [12] int arrays (6 white, 6 black
 * pieces)
 * @return true if successful, false otherwise
 */
bool cuda_batch_count_pieces(const std::vector<std::string> &fens,
                             std::vector<std::vector<int>> &piece_counts);

/**
 * Batch position hashing on GPU
 * Generates hash values for transposition table lookups
 *
 * @param fens Vector of FEN strings
 * @param hashes Output vector of hash values
 * @return true if successful, false otherwise
 */
bool cuda_batch_hash_positions(const std::vector<std::string> &fens,
                               std::vector<unsigned long long> &hashes);

// Python-friendly wrappers that return values instead of using output
// parameters
std::vector<int> cuda_batch_evaluate_py(const std::vector<std::string> &fens);
std::vector<std::vector<int>>
cuda_batch_count_pieces_py(const std::vector<std::string> &fens);
std::vector<unsigned long long>
cuda_batch_hash_positions_py(const std::vector<std::string> &fens);

#endif // CUDA_EVAL_H
