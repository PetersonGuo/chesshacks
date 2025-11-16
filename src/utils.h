#ifndef UTILS_H
#define UTILS_H

#include "chess_board.h"
#include <string>

// Helper function to parse SAN (Standard Algebraic Notation) move
// Returns true if move was successfully parsed and applied
bool parse_and_apply_san(ChessBoard &board, const std::string &san_move);

// Convert PGN string to FEN string
std::string pgn_to_fen(const std::string &pgn);

#endif // UTILS_H
