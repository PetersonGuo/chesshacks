#ifndef EVALUATION_H
#define EVALUATION_H

#include "chess_board.h"
#include "move_ordering.h"
#include "transposition_table.h"
#include <string>
#include <vector>

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
                 HistoryTable *history = nullptr);

// Batch evaluate multiple positions in parallel
std::vector<int> batch_evaluate_mt(const std::vector<std::string> &fens,
                                    int num_threads = 0);

#endif // EVALUATION_H
