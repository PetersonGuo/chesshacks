#ifndef EVALUATION_H
#define EVALUATION_H

#include "chess_board.h"
#include "move_ordering.h"
#include "transposition_table.h"
#include "nnue_evaluator.h"
#include <string>
#include <vector>
#include <memory>

// Piece-Square Tables for positional evaluation
namespace PieceSquareTables {
extern const int pawn_table[64];
extern const int knight_table[64];
extern const int bishop_table[64];
extern const int rook_table[64];
extern const int queen_table[64];
extern const int king_middlegame_table[64];
} // namespace PieceSquareTables

// Get piece-square table value for a piece at a given square
int get_piece_square_value(Piece piece, int square);

// Get piece value (for MVV-LVA)
int get_piece_value(Piece piece);

// Enhanced evaluation with piece-square tables
int evaluate_with_pst(const std::string &fen);

// MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score
int mvv_lva_score(const ChessBoard &board, const Move &move);

// Static Exchange Evaluation (SEE)
int static_exchange_eval(ChessBoard &board, const Move &move);

// Order moves for better alpha-beta pruning
void order_moves(ChessBoard &board, std::vector<Move> &moves,
                 TranspositionTable *tt, const std::string &fen,
                 KillerMoves *killers = nullptr, int ply = 0,
                 HistoryTable *history = nullptr,
                 CounterMoveTable *counters = nullptr, int prev_piece = 0,
                 int prev_to = -1, ContinuationHistory *cont_history = nullptr);

// Batch evaluate multiple positions in parallel
std::vector<int> batch_evaluate_mt(const std::vector<std::string> &fens,
                                    int num_threads = 0);

// ============================================================================
// NNUE EVALUATION
// ============================================================================

// Global NNUE evaluator instance
extern std::unique_ptr<NNUEEvaluator> g_nnue_evaluator;

// Initialize NNUE evaluator with model file
// Returns true if loaded successfully, false otherwise
bool init_nnue(const std::string& model_path);

// Check if NNUE is loaded and ready
bool is_nnue_loaded();

// Evaluate position using NNUE (if loaded)
// Falls back to PST evaluation if NNUE is not loaded
int evaluate_nnue(const std::string& fen);
int evaluate_nnue(const ChessBoard& board);

// Main evaluation function that uses NNUE if available, otherwise PST
int evaluate(const std::string& fen);
int evaluate(const ChessBoard& board);

#endif // EVALUATION_H
