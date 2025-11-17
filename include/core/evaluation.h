#ifndef EVALUATION_H
#define EVALUATION_H

#include "bitboard/bitboard_state.h"
#include "move_ordering.h"
#include "nnue_evaluator.h"

#include <cstdint>
#include <memory>
#include <vector>

// Get piece value (for MVV-LVA)
int get_piece_value(BoardPiece piece);

// Simple material-only evaluation (moved from Python)
int evaluate_material(const bitboard::BitboardState &board);

// MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score
int mvv_lva_score(const bitboard::BitboardState &board, const BoardMove &move);

// Static Exchange Evaluation (SEE)
int static_exchange_eval(bitboard::BitboardState &board, const BoardMove &move);

// Order moves for better alpha-beta pruning
void order_moves(bitboard::BitboardState &board, std::vector<BoardMove> &moves,
                 uint16_t tt_best_move = 0, KillerMoves *killers = nullptr,
                 int ply = 0, HistoryTable *history = nullptr,
                 CounterMoveTable *counters = nullptr, int prev_piece = 0,
                 int prev_to = -1, ContinuationHistory *cont_history = nullptr);

// Batch evaluate multiple positions in parallel
std::vector<int>
batch_evaluate_mt(const std::vector<bitboard::BitboardState> &boards,
                  int num_threads = 0);

// ============================================================================
// NNUE EVALUATION
// ============================================================================

// Global NNUE evaluator instance
extern std::unique_ptr<NNUEEvaluator> g_nnue_evaluator;

// Initialize NNUE evaluator with model file
// Returns true if loaded successfully, false otherwise
bool init_nnue(const std::string &model_path);

// Check if NNUE is loaded and ready
bool is_nnue_loaded();

// Evaluate position using NNUE (if loaded)
// Falls back to material evaluation if NNUE is not loaded
int evaluate_nnue(const bitboard::BitboardState &board);

// Main evaluation function that uses NNUE if available, otherwise material
int evaluate(const bitboard::BitboardState &board);

#endif // EVALUATION_H
