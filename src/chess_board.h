#ifndef CHESS_BOARD_H
#define CHESS_BOARD_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Piece types
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
  int from;
  int to;
  Piece promotion;

  Move(int f, int t, Piece p = EMPTY) : from(f), to(t), promotion(p) {}
};

class ChessBoard {
public:
  ChessBoard();
  explicit ChessBoard(const std::string &fen);

  // Parse and generate FEN
  void from_fen(const std::string &fen);
  std::string to_fen() const;

  // Move generation
  std::vector<Move> generate_legal_moves() const;
  void make_move(const Move &move);
  void unmake_move(const Move &move);

  // Board state
  bool is_white_to_move() const { return white_to_move; }
  bool is_check() const;
  bool is_checkmate() const;
  bool is_stalemate() const;

  // Piece access for move ordering
  Piece get_piece_at(int square) const { return board[square]; }
  bool is_capture(const Move &move) const { return board[move.to] != EMPTY; }

private:
  std::array<Piece, 64> board;
  bool white_to_move;
  uint8_t castling_rights; // KQkq bits
  int en_passant_square;
  int halfmove_clock;
  int fullmove_number;

  // Move generation helpers
  void generate_pawn_moves(int square, std::vector<Move> &moves) const;
  void generate_knight_moves(int square, std::vector<Move> &moves) const;
  void generate_bishop_moves(int square, std::vector<Move> &moves) const;
  void generate_rook_moves(int square, std::vector<Move> &moves) const;
  void generate_queen_moves(int square, std::vector<Move> &moves) const;
  void generate_king_moves(int square, std::vector<Move> &moves) const;
  void
  generate_sliding_moves(int square,
                         const std::vector<std::pair<int, int>> &directions,
                         std::vector<Move> &moves) const;

  // Board state helpers
  bool is_square_attacked(int square, bool by_white) const;
  int find_king(bool white) const;
  bool is_valid_square(int rank, int file) const;
  int to_square(int rank, int file) const;

  // Piece saved for unmake
  struct BoardState {
    Piece captured_piece;
    uint8_t castling_rights;
    int en_passant_square;
    int halfmove_clock;
  };
  std::vector<BoardState> state_stack;
};

#endif // CHESS_BOARD_H
