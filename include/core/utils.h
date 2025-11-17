#ifndef UTILS_H
#define UTILS_H

#include "bitboard/bitboard_state.h"
#include <string>

// Helper function to parse SAN (Standard Algebraic Notation) move
// Returns true if move was successfully parsed and applied
bool parse_and_apply_san(bitboard::BitboardState &board,
                         const std::string &san_move);

// Convert PGN string to FEN string
std::string pgn_to_fen(const std::string &pgn);

// Convert PGN string directly to a BitboardState
bitboard::BitboardState pgn_to_bitboard(const std::string &pgn);

#endif // UTILS_H
