#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../../third_party/virgo/virgo.h"

enum Piece : int8_t {
  EMPTY = 0,
  W_PAWN = 1,
  W_KNIGHT = 2,
  W_BISHOP = 3,
  W_ROOK = 4,
  W_QUEEN = 5,
  W_KING = 6,
  B_PAWN = -1,
  B_KNIGHT = -2,
  B_BISHOP = -3,
  B_ROOK = -4,
  B_QUEEN = -5,
  B_KING = -6
};

struct Move {
  int from = 0;
  int to = 0;
  Piece promotion = EMPTY;
  uint16_t encoded = 0;

  Move() = default;
  Move(int f, int t, Piece p = EMPTY, uint16_t e = 0)
      : from(f), to(t), promotion(p), encoded(e) {}
};

namespace bitboard {

class BitboardState {
public:
  BitboardState();
  explicit BitboardState(const std::string &fen);

  void from_fen(const std::string &fen);
  void set_from_fen(const std::string &fen);
  std::string to_fen() const;

  std::vector<Move> generate_legal_moves() const;
  void make_move(const Move &move);
  void unmake_move(const Move &move);

  bool white_to_move() const;
  bool is_check() const;
  bool is_checkmate() const;
  bool is_stalemate() const;

  Piece get_piece_at(int square) const;
  bool is_capture(const Move &move) const;

  uint16_t ensure_encoded(const Move &move) const;
  Move decode_move(uint16_t encoded) const;
  std::string move_to_uci(uint16_t encoded) const;
  uint64_t zobrist() const;
  void set_from_components(const std::array<int8_t, 64> &encoded,
                           bool white_to_move, const std::string &castling,
                           int en_passant_square, int halfmove_clock,
                           int fullmove_number);
  static BitboardState
  from_components(const std::array<int8_t, 64> &encoded, bool white_to_move,
                  const std::string &castling, int en_passant_square,
                  int halfmove_clock, int fullmove_number);

  uint64_t occupancy() const;

  const virgo::Chessboard &native() const { return board_; }
  virgo::Chessboard &native() { return board_; }

private:
  static void ensure_initialized();

  Move decode_move(uint16_t encoded, virgo::Player mover) const;
  Piece convert_piece(const std::pair<virgo::Piece, virgo::Player> &cell) const;
  char piece_to_char(Piece piece) const;
  std::string castling_string(const virgo::Chessboard &board) const;
  std::string en_passant_string(const virgo::Chessboard &board) const;
  int parse_fullmove_number(const std::string &fen) const;

  virgo::Chessboard board_;
  int fullmove_number_ = 1;
};

} // namespace bitboard

