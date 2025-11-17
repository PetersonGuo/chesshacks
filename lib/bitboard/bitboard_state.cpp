#include "bitboard/bitboard_state.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>

namespace bitboard {
namespace {

constexpr const char *kStartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct FenParts {
  std::string board;
  std::string active;
  std::string castling;
  std::string en_passant;
  int halfmove = 0;
  int fullmove = 1;
};

FenParts ParseFen(const std::string &fen) {
  FenParts parts;
  std::istringstream iss(fen);
  if (!(iss >> parts.board >> parts.active >> parts.castling >>
        parts.en_passant >> parts.halfmove >> parts.fullmove)) {
    throw std::invalid_argument("Invalid FEN: " + fen);
  }
  if (parts.castling.empty()) {
    parts.castling = "-";
  }
  if (parts.en_passant.empty()) {
    parts.en_passant = "-";
  }
  return parts;
}

bool IsPromotionFlag(::MoveFlags flag) {
  switch (flag) {
  case PR_KNIGHT:
  case PR_BISHOP:
  case PR_ROOK:
  case PR_QUEEN:
  case PC_KNIGHT:
  case PC_BISHOP:
  case PC_ROOK:
  case PC_QUEEN:
    return true;
  default:
    return false;
  }
}

BoardPiece PromotionFromFlag(::MoveFlags flag, bool is_white) {
  const BoardPiece knight = is_white ? W_KNIGHT : B_KNIGHT;
  const BoardPiece bishop = is_white ? W_BISHOP : B_BISHOP;
  const BoardPiece rook = is_white ? W_ROOK : B_ROOK;
  const BoardPiece queen = is_white ? W_QUEEN : B_QUEEN;
  switch (flag) {
  case PR_KNIGHT:
  case PC_KNIGHT:
    return knight;
  case PR_BISHOP:
  case PC_BISHOP:
    return bishop;
  case PR_ROOK:
  case PC_ROOK:
    return rook;
  case PR_QUEEN:
  case PC_QUEEN:
    return queen;
  default:
    return EMPTY;
  }
}

std::string SquareToString(::Square square) {
  if (square == ::NO_SQUARE) {
    return "-";
  }
  int value = static_cast<int>(square);
  char file = static_cast<char>('a' + (value % 8));
  char rank = static_cast<char>('1' + (value / 8));
  return std::string{file, rank};
}

void InitSurge() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    initialise_all_databases();
    zobrist::initialise_zobrist_keys();
  });
}

} // namespace

void BitboardState::ensure_initialized() { InitSurge(); }

BitboardState::BitboardState() {
  ensure_initialized();
  reset_board();
  set_from_fen(kStartFEN);
}

BitboardState::BitboardState(const std::string &fen) {
  ensure_initialized();
  reset_board();
  set_from_fen(fen);
}

BitboardState::BitboardState(const BitboardState &other) {
  ensure_initialized();
  new (&board_)::Position(other.board_);
  halfmove_clock_ = other.halfmove_clock_;
  fullmove_number_ = other.fullmove_number_;
  halfmove_history_ = other.halfmove_history_;
}

BitboardState &BitboardState::operator=(const BitboardState &other) {
  if (this == &other) {
    return *this;
  }
  ensure_initialized();
  board_.~Position();
  new (&board_)::Position(other.board_);
  halfmove_clock_ = other.halfmove_clock_;
  fullmove_number_ = other.fullmove_number_;
  halfmove_history_ = other.halfmove_history_;
  return *this;
}

BitboardState::BitboardState(BitboardState &&other) noexcept {
  ensure_initialized();
  new (&board_)::Position(other.board_);
  halfmove_clock_ = other.halfmove_clock_;
  fullmove_number_ = other.fullmove_number_;
  halfmove_history_ = std::move(other.halfmove_history_);
}

BitboardState &BitboardState::operator=(BitboardState &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  ensure_initialized();
  board_.~Position();
  new (&board_)::Position(other.board_);
  halfmove_clock_ = other.halfmove_clock_;
  fullmove_number_ = other.fullmove_number_;
  halfmove_history_ = std::move(other.halfmove_history_);
  return *this;
}

void BitboardState::reset_board() {
  board_.~Position();
  new (&board_)::Position();
  halfmove_clock_ = 0;
  fullmove_number_ = 1;
  halfmove_history_.clear();
}

::Position &BitboardState::board() { return board_; }
::Position &BitboardState::board() const { return board_; }

int BitboardState::current_ply() const { return board().ply(); }

void BitboardState::from_fen(const std::string &fen) { set_from_fen(fen); }

void BitboardState::set_from_fen(const std::string &fen) {
  ensure_initialized();
  apply_fen(fen);
}

void BitboardState::apply_fen(const std::string &fen) {
  FenParts parts = ParseFen(fen);
  reset_board();
  std::string minimal = parts.board + " " + parts.active + " " + parts.castling;
  ::Position::set(minimal, board());
  set_en_passant_square(parts.en_passant);
  set_halfmove_clock(parts.halfmove);
  set_fullmove_number(parts.fullmove);
}

void BitboardState::set_en_passant_square(const std::string &ep) {
  auto &undo = board().history[current_ply()];
  if (ep == "-" || ep.empty()) {
    undo.epsq = ::NO_SQUARE;
    return;
  }
  if (ep.size() != 2 || ep[0] < 'a' || ep[0] > 'h' || ep[1] < '1' ||
      ep[1] > '8') {
    throw std::invalid_argument("Invalid en passant square: " + ep);
  }
  int file = ep[0] - 'a';
  int rank = ep[1] - '1';
  undo.epsq = static_cast<::Square>(rank * 8 + file);
}

void BitboardState::set_halfmove_clock(int value) {
  halfmove_clock_ = std::max(0, value);
  halfmove_history_.clear();
}

void BitboardState::set_fullmove_number(int value) {
  fullmove_number_ = std::max(1, value);
}

std::string BitboardState::to_fen() const {
  std::ostringstream oss;
  oss << board_string() << ' ' << (white_to_move() ? 'w' : 'b') << ' '
      << castling_string() << ' ' << en_passant_string() << ' '
      << halfmove_clock_ << ' ' << fullmove_number_;
  return oss.str();
}

std::string BitboardState::board_string() const {
  std::ostringstream oss;
  for (int rank = 7; rank >= 0; --rank) {
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      int square = rank * 8 + file;
      BoardPiece piece =
          convert_piece(board().at(static_cast<::Square>(square)));
      if (piece == EMPTY) {
        ++empty;
        continue;
      }
      if (empty != 0) {
        oss << empty;
        empty = 0;
      }
      oss << piece_to_char(piece);
    }
    if (empty != 0) {
      oss << empty;
    }
    if (rank > 0) {
      oss << '/';
    }
  }
  return oss.str();
}

std::string BitboardState::castling_string() const {
  const auto entry = board().history[current_ply()].entry;
  std::string rights;
  if (!(entry & WHITE_OO_MASK)) {
    rights.push_back('K');
  }
  if (!(entry & WHITE_OOO_MASK)) {
    rights.push_back('Q');
  }
  if (!(entry & BLACK_OO_MASK)) {
    rights.push_back('k');
  }
  if (!(entry & BLACK_OOO_MASK)) {
    rights.push_back('q');
  }
  if (rights.empty()) {
    return "-";
  }
  return rights;
}

std::string BitboardState::en_passant_string() const {
  return SquareToString(board().history[current_ply()].epsq);
}

std::vector<BoardMove> BitboardState::generate_legal_moves() const {
  std::array<::Move, 256> buffer{};
  ::Move *end = nullptr;
  bool mover_white = board().turn() == WHITE;
  if (mover_white) {
    end = board().generate_legals<WHITE>(buffer.data());
  } else {
    end = board().generate_legals<BLACK>(buffer.data());
  }

  std::vector<BoardMove> moves;
  moves.reserve(static_cast<size_t>(end - buffer.data()));
  for (auto it = buffer.data(); it != end; ++it) {
    moves.push_back(decode_surge_move(*it, mover_white));
  }
  return moves;
}

BoardMove BitboardState::decode_surge_move(const ::Move &surge_move,
                                           bool mover_white) const {
  int from = static_cast<int>(surge_move.from());
  int to = static_cast<int>(surge_move.to());
  ::MoveFlags flags = surge_move.flags();
  BoardPiece promotion =
      IsPromotionFlag(flags) ? PromotionFromFlag(flags, mover_white) : EMPTY;
  uint16_t encoded = static_cast<uint16_t>(
      (static_cast<uint16_t>(flags) << 12) | (from << 6) | to);
  return BoardMove(from, to, promotion, encoded);
}

void BitboardState::make_move(const BoardMove &move) {
  uint16_t encoded = move.encoded ? move.encoded : ensure_encoded(move);
  ::Move surge_move(encoded);
  bool mover_white = white_to_move();
  auto surge_piece = board().at(static_cast<::Square>(move.from));
  bool pawn_move = surge_piece == WHITE_PAWN || surge_piece == BLACK_PAWN ||
                   surge_flag_is_pawn_move(surge_move.flags());
  bool capture = surge_flag_is_capture(surge_move.flags());

  halfmove_history_.push_back(halfmove_clock_);
  if (pawn_move || capture) {
    halfmove_clock_ = 0;
  } else {
    ++halfmove_clock_;
  }

  if (mover_white) {
    board().play<WHITE>(surge_move);
  } else {
    board().play<BLACK>(surge_move);
    ++fullmove_number_;
  }
}

void BitboardState::unmake_move(const BoardMove &move) {
  if (halfmove_history_.empty()) {
    throw std::runtime_error("Cannot unmake move without history");
  }
  bool next_to_move_is_white = white_to_move();
  uint16_t encoded = move.encoded ? move.encoded : ensure_encoded(move);
  ::Move surge_move(encoded);

  if (next_to_move_is_white) {
    board().undo<BLACK>(surge_move);
    if (fullmove_number_ > 1) {
      --fullmove_number_;
    }
  } else {
    board().undo<WHITE>(surge_move);
  }
  halfmove_clock_ = halfmove_history_.back();
  halfmove_history_.pop_back();
}

bool BitboardState::white_to_move() const { return board().turn() == WHITE; }

bool BitboardState::is_check() const {
  if (white_to_move()) {
    return board().in_check<WHITE>();
  }
  return board().in_check<BLACK>();
}

bool BitboardState::is_checkmate() const {
  auto moves = generate_legal_moves();
  return moves.empty() && is_check();
}

bool BitboardState::is_stalemate() const {
  auto moves = generate_legal_moves();
  return moves.empty() && !is_check();
}

BoardPiece BitboardState::get_piece_at(int square) const {
  if (square < 0 || square >= 64) {
    throw std::out_of_range("Square out of bounds");
  }
  return convert_piece(board().at(static_cast<::Square>(square)));
}

bool BitboardState::is_capture(const BoardMove &move) const {
  uint16_t encoded = move.encoded ? move.encoded : ensure_encoded(move);
  ::MoveFlags flag =
      static_cast<::MoveFlags>((encoded >> 12) & static_cast<uint16_t>(0xF));
  return surge_flag_is_capture(flag);
}

uint16_t BitboardState::ensure_encoded(const BoardMove &move) const {
  auto legal_moves = generate_legal_moves();
  for (const auto &candidate : legal_moves) {
    if (candidate.from == move.from && candidate.to == move.to) {
      if (move.promotion == EMPTY || candidate.promotion == move.promotion) {
        return candidate.encoded;
      }
    }
  }
  throw std::runtime_error("Requested move is not legal in current position");
}

BoardMove BitboardState::decode_move(uint16_t encoded) const {
  return decode_move(encoded, white_to_move());
}

BoardMove BitboardState::decode_move(uint16_t encoded, bool mover_white) const {
  int from = (encoded >> 6) & 0x3F;
  int to = encoded & 0x3F;
  ::MoveFlags flag =
      static_cast<::MoveFlags>((encoded >> 12) & static_cast<uint16_t>(0xF));
  BoardPiece promotion =
      IsPromotionFlag(flag) ? PromotionFromFlag(flag, mover_white) : EMPTY;
  return BoardMove(from, to, promotion, encoded);
}

std::string BitboardState::move_to_uci(uint16_t encoded) const {
  BoardMove move = decode_move(encoded);
  std::string uci;
  uci.reserve(5);
  uci.push_back(static_cast<char>('a' + (move.from % 8)));
  uci.push_back(static_cast<char>('1' + (move.from / 8)));
  uci.push_back(static_cast<char>('a' + (move.to % 8)));
  uci.push_back(static_cast<char>('1' + (move.to / 8)));
  if (move.promotion != EMPTY) {
    char promo = 'q';
    switch (std::abs(static_cast<int>(move.promotion))) {
    case 2:
      promo = 'n';
      break;
    case 3:
      promo = 'b';
      break;
    case 4:
      promo = 'r';
      break;
    case 5:
      promo = 'q';
      break;
    default:
      promo = 'q';
      break;
    }
    uci.push_back(promo);
  }
  return uci;
}

uint64_t BitboardState::zobrist() const { return board().get_hash(); }

void BitboardState::set_from_components(const std::array<int8_t, 64> &encoded,
                                        bool white, const std::string &castling,
                                        int en_passant_square,
                                        int halfmove_clock,
                                        int fullmove_number) {
  std::ostringstream board_part;
  for (int rank = 7; rank >= 0; --rank) {
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      int idx = rank * 8 + file;
      BoardPiece piece = static_cast<BoardPiece>(encoded[idx]);
      if (piece == EMPTY) {
        ++empty;
        continue;
      }
      if (empty != 0) {
        board_part << empty;
        empty = 0;
      }
      board_part << piece_to_char(piece);
    }
    if (empty != 0) {
      board_part << empty;
    }
    if (rank > 0) {
      board_part << '/';
    }
  }
  std::string castling_part = castling.empty() ? "-" : castling;
  std::string ep = "-";
  if (en_passant_square >= 0 && en_passant_square < 64) {
    char file = static_cast<char>('a' + (en_passant_square % 8));
    char rank = static_cast<char>('1' + (en_passant_square / 8));
    ep = std::string{file, rank};
  }
  std::ostringstream fen;
  fen << board_part.str() << ' ' << (white ? 'w' : 'b') << ' ' << castling_part
      << ' ' << ep << ' ' << halfmove_clock << ' ' << fullmove_number;
  set_from_fen(fen.str());
}

BitboardState
BitboardState::from_components(const std::array<int8_t, 64> &encoded,
                               bool white, const std::string &castling,
                               int en_passant_square, int halfmove_clock,
                               int fullmove_number) {
  BitboardState state;
  state.set_from_components(encoded, white, castling, en_passant_square,
                            halfmove_clock, fullmove_number);
  return state;
}

uint64_t BitboardState::occupancy() const {
  return board().all_pieces<WHITE>() | board().all_pieces<BLACK>();
}

BoardPiece BitboardState::convert_piece(::Piece piece) const {
  switch (piece) {
  case WHITE_PAWN:
    return W_PAWN;
  case WHITE_KNIGHT:
    return W_KNIGHT;
  case WHITE_BISHOP:
    return W_BISHOP;
  case WHITE_ROOK:
    return W_ROOK;
  case WHITE_QUEEN:
    return W_QUEEN;
  case WHITE_KING:
    return W_KING;
  case BLACK_PAWN:
    return B_PAWN;
  case BLACK_KNIGHT:
    return B_KNIGHT;
  case BLACK_BISHOP:
    return B_BISHOP;
  case BLACK_ROOK:
    return B_ROOK;
  case BLACK_QUEEN:
    return B_QUEEN;
  case BLACK_KING:
    return B_KING;
  default:
    return EMPTY;
  }
}

char BitboardState::piece_to_char(BoardPiece piece) const {
  int abs_piece = piece < 0 ? -piece : piece;
  char symbol = '?';
  switch (abs_piece) {
  case 1:
    symbol = 'P';
    break;
  case 2:
    symbol = 'N';
    break;
  case 3:
    symbol = 'B';
    break;
  case 4:
    symbol = 'R';
    break;
  case 5:
    symbol = 'Q';
    break;
  case 6:
    symbol = 'K';
    break;
  default:
    symbol = '?';
    break;
  }
  if (piece < 0) {
    symbol = static_cast<char>(std::tolower(symbol));
  }
  return symbol;
}

bool BitboardState::surge_flag_is_capture(::MoveFlags flag) const {
  switch (flag) {
  case CAPTURE:
  case EN_PASSANT:
  case PC_KNIGHT:
  case PC_BISHOP:
  case PC_ROOK:
  case PC_QUEEN:
    return true;
  default:
    return false;
  }
}

bool BitboardState::surge_flag_is_pawn_move(::MoveFlags flag) const {
  switch (flag) {
  case DOUBLE_PUSH:
  case EN_PASSANT:
  case PR_KNIGHT:
  case PR_BISHOP:
  case PR_ROOK:
  case PR_QUEEN:
  case PC_KNIGHT:
  case PC_BISHOP:
  case PC_ROOK:
  case PC_QUEEN:
    return true;
  default:
    return false;
  }
}

} // namespace bitboard
