#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../../third_party/surge/src/position.h"

enum BoardPiece : int8_t {
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

struct BoardMove {
  int from = 0;
  int to = 0;
  BoardPiece promotion = EMPTY;
  uint16_t encoded = 0;

  BoardMove() = default;
  BoardMove(int f, int t, BoardPiece p = EMPTY, uint16_t e = 0)
      : from(f), to(t), promotion(p), encoded(e) {}
};

namespace bitboard {

class BitboardState {
public:
  BitboardState();
  explicit BitboardState(const std::string &fen);
  BitboardState(const BitboardState &other);
  BitboardState &operator=(const BitboardState &other);
  BitboardState(BitboardState &&other) noexcept;
  BitboardState &operator=(BitboardState &&other) noexcept;

  void from_fen(const std::string &fen);
  void set_from_fen(const std::string &fen);
  std::string to_fen() const;

  std::vector<BoardMove> generate_legal_moves() const;
  void make_move(const BoardMove &move);
  void unmake_move(const BoardMove &move);

  bool white_to_move() const;
  bool is_check() const;
  bool is_checkmate() const;
  bool is_stalemate() const;

  BoardPiece get_piece_at(int square) const;
  bool is_capture(const BoardMove &move) const;

  uint16_t ensure_encoded(const BoardMove &move) const;
  BoardMove decode_move(uint16_t encoded) const;
  std::string move_to_uci(uint16_t encoded) const;
  uint64_t zobrist() const;
  void set_from_components(const std::array<int8_t, 64> &encoded,
                           bool white_to_move, const std::string &castling,
                           int en_passant_square, int halfmove_clock,
                           int fullmove_number);
  static BitboardState from_components(const std::array<int8_t, 64> &encoded,
                                       bool white_to_move,
                                       const std::string &castling,
                                       int en_passant_square,
                                       int halfmove_clock, int fullmove_number);

  uint64_t occupancy() const;

private:
  static void ensure_initialized();
  void reset_board();
  ::Position &board();
  ::Position &board() const;
  int current_ply() const;
  void set_en_passant_square(const std::string &ep);
  std::string en_passant_string() const;
  std::string castling_string() const;
  std::string board_string() const;
  BoardPiece convert_piece(::Piece piece) const;
  BoardMove decode_move(uint16_t encoded, bool mover_white) const;
  BoardMove decode_surge_move(const ::Move &surge_move, bool mover_white) const;
  bool surge_flag_is_capture(::MoveFlags flag) const;
  bool surge_flag_is_pawn_move(::MoveFlags flag) const;
  char piece_to_char(BoardPiece piece) const;
  void apply_fen(const std::string &fen);
  void set_halfmove_clock(int value);
  void set_fullmove_number(int value);

  mutable ::Position board_;
  int halfmove_clock_ = 0;
  int fullmove_number_ = 1;
  std::vector<int> halfmove_history_;
};

} // namespace bitboard
