#include "bitboard_state.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bitboard {

namespace {

void init_virgo_once() {
  static std::once_flag flag;
  std::call_once(flag, []() { virgo::virgo_init(); });
}

constexpr const char *kStartFEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

uint8_t CastlingMaskFromString(const std::string &castling) {
  if (castling.empty() || castling == "-") {
    return 0;
  }
  uint8_t mask = 0;
  for (char c : castling) {
    switch (c) {
    case 'K':
      mask |= 0x08;
      break;
    case 'Q':
      mask |= 0x04;
      break;
    case 'k':
      mask |= 0x02;
      break;
    case 'q':
      mask |= 0x01;
      break;
    default:
      break;
    }
  }
  return mask;
}

uint64_t Random64() {
  static std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
  return rng();
}

std::array<std::array<uint64_t, 64>, 12> g_piece_keys;
uint64_t g_side_key;
std::array<uint64_t, 16> g_castle_keys;
std::array<uint64_t, 8> g_en_passant_keys;

void init_zobrist_once() {
  static std::once_flag zobrist_flag;
  std::call_once(zobrist_flag, []() {
    for (auto &piece_array : g_piece_keys) {
      for (auto &entry : piece_array) {
        entry = Random64();
      }
    }
    g_side_key = Random64();
    for (auto &entry : g_castle_keys) {
      entry = Random64();
    }
    for (auto &entry : g_en_passant_keys) {
      entry = Random64();
    }
  });
}

int piece_index(Piece piece) {
  if (piece > 0) {
    return static_cast<int>(piece) - 1;
  } else if (piece < 0) {
    return 6 + (-static_cast<int>(piece) - 1);
  }
  return -1;
}

Piece PromotionFromType(virgo::MoveType type, virgo::Player mover) {
  bool white = mover == virgo::WHITE;
  switch (type) {
  case virgo::PQ_Q:
  case virgo::PC_Q:
    return white ? W_QUEEN : B_QUEEN;
  case virgo::PQ_R:
  case virgo::PC_R:
    return white ? W_ROOK : B_ROOK;
  case virgo::PQ_B:
  case virgo::PC_B:
    return white ? W_BISHOP : B_BISHOP;
  case virgo::PQ_N:
  case virgo::PC_N:
    return white ? W_KNIGHT : B_KNIGHT;
  default:
    return EMPTY;
  }
}

bool IsCaptureType(virgo::MoveType type) {
  switch (type) {
  case virgo::CAPTURE:
  case virgo::EN_PASSANT:
  case virgo::PC_B:
  case virgo::PC_R:
  case virgo::PC_N:
  case virgo::PC_Q:
    return true;
  default:
    return false;
  }
}

inline bool IsValidSquare(int rank, int file) {
  return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

inline int ToSquare(int rank, int file) { return rank * 8 + file; }

bool SquareAttacked(const std::array<Piece, 64> &board, int square,
                    bool by_white) {
  int rank = square / 8;
  int file = square % 8;
  int direction = by_white ? 1 : -1;

  auto piece_at = [&](int r, int f) -> Piece {
    if (!IsValidSquare(r, f))
      return EMPTY;
    return board[ToSquare(r, f)];
  };

  // Pawn attacks
  if (IsValidSquare(rank - direction, file - 1) &&
      piece_at(rank - direction, file - 1) == (by_white ? W_PAWN : B_PAWN)) {
    return true;
  }
  if (IsValidSquare(rank - direction, file + 1) &&
      piece_at(rank - direction, file + 1) == (by_white ? W_PAWN : B_PAWN)) {
    return true;
  }

  // Knights
  static const std::pair<int, int> knight_offsets[] = {
      {2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
  for (const auto &offset : knight_offsets) {
    int nr = rank + offset.first;
    int nf = file + offset.second;
    if (IsValidSquare(nr, nf) &&
        piece_at(nr, nf) == (by_white ? W_KNIGHT : B_KNIGHT)) {
      return true;
    }
  }

  // Kings
  for (int dr = -1; dr <= 1; ++dr) {
    for (int df = -1; df <= 1; ++df) {
      if (dr == 0 && df == 0)
        continue;
      int nr = rank + dr;
      int nf = file + df;
      if (IsValidSquare(nr, nf) &&
          piece_at(nr, nf) == (by_white ? W_KING : B_KING)) {
        return true;
      }
    }
  }

  auto sliding_attacked = [&](const std::pair<int, int> *dirs, size_t count,
                              Piece primary, Piece queen) {
    for (size_t i = 0; i < count; ++i) {
      int nr = rank + dirs[i].first;
      int nf = file + dirs[i].second;
      while (IsValidSquare(nr, nf)) {
        Piece occupant = piece_at(nr, nf);
        if (occupant != EMPTY) {
          if (occupant == primary || occupant == queen) {
            return true;
          }
          break;
        }
        nr += dirs[i].first;
        nf += dirs[i].second;
      }
    }
    return false;
  };

  static const std::pair<int, int> bishop_dirs[] = {
      {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  static const std::pair<int, int> rook_dirs[] = {
      {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  if (sliding_attacked(bishop_dirs, 4, by_white ? W_BISHOP : B_BISHOP,
                       by_white ? W_QUEEN : B_QUEEN)) {
    return true;
  }

  if (sliding_attacked(rook_dirs, 4, by_white ? W_ROOK : B_ROOK,
                       by_white ? W_QUEEN : B_QUEEN)) {
    return true;
  }

  return false;
}

} // namespace

void BitboardState::ensure_initialized() {
  init_virgo_once();
  init_zobrist_once();
}

BitboardState::BitboardState() { from_fen(kStartFEN); }

BitboardState::BitboardState(const std::string &fen) { from_fen(fen); }

void BitboardState::from_fen(const std::string &fen) { set_from_fen(fen); }

void BitboardState::set_from_fen(const std::string &fen) {
  ensure_initialized();
  board_ = virgo::position_from_fen(fen);
  fullmove_number_ = parse_fullmove_number(fen);
  if (fullmove_number_ < 1) {
    fullmove_number_ = 1;
  }
}

std::string BitboardState::to_fen() const {
  std::ostringstream oss;
  const virgo::Chessboard &native = board_;

  for (int rank = 7; rank >= 0; --rank) {
    int empty_count = 0;
    for (int file = 0; file < 8; ++file) {
      int square = rank * 8 + file;
      Piece piece = convert_piece(native[square]);
      if (piece == EMPTY) {
        ++empty_count;
        continue;
      }
      if (empty_count > 0) {
        oss << empty_count;
        empty_count = 0;
      }
      oss << piece_to_char(piece);
    }
    if (empty_count > 0) {
      oss << empty_count;
    }
    if (rank > 0) {
      oss << '/';
    }
  }

  oss << ' ' << (white_to_move() ? 'w' : 'b') << ' ';
  oss << castling_string(native) << ' ';
  oss << en_passant_string(native) << ' ';
  oss << static_cast<int>(native.get_halfmove_clock()) << ' '
      << fullmove_number_;
  return oss.str();
}

std::vector<Move> BitboardState::generate_legal_moves() const {
  std::vector<uint16_t> encoded_moves;
  encoded_moves.reserve(128);
  virgo::Player mover = white_to_move() ? virgo::WHITE : virgo::BLACK;
  if (mover == virgo::WHITE) {
    virgo::get_legal_moves<virgo::WHITE>(
        const_cast<virgo::Chessboard &>(board_), encoded_moves);
  } else {
    virgo::get_legal_moves<virgo::BLACK>(
        const_cast<virgo::Chessboard &>(board_), encoded_moves);
  }
  std::vector<Move> moves;
  moves.reserve(encoded_moves.size());
  for (uint16_t encoded : encoded_moves) {
    moves.push_back(decode_move(encoded, mover));
  }
  return moves;
}

void BitboardState::make_move(const Move &move) {
  uint16_t encoded = move.encoded ? move.encoded : ensure_encoded(move);
  if (white_to_move()) {
    virgo::make_move<virgo::WHITE>(encoded, board_);
  } else {
    virgo::make_move<virgo::BLACK>(encoded, board_);
    ++fullmove_number_;
  }
}

void BitboardState::unmake_move(const Move &) {
  virgo::Player last_player = white_to_move() ? virgo::BLACK : virgo::WHITE;
  if (last_player == virgo::WHITE) {
    virgo::take_move<virgo::WHITE>(board_);
  } else {
    virgo::take_move<virgo::BLACK>(board_);
    if (fullmove_number_ > 1) {
      --fullmove_number_;
    }
  }
}

bool BitboardState::white_to_move() const {
  return const_cast<virgo::Chessboard &>(board_).get_next_to_move() ==
         virgo::WHITE;
}

bool BitboardState::is_check() const {
  virgo::Player player = white_to_move() ? virgo::WHITE : virgo::BLACK;
  bool enemy_is_white = (player == virgo::BLACK);

  std::array<Piece, 64> snapshot;
  const virgo::Chessboard &native = board_;
  for (int sq = 0; sq < 64; ++sq) {
    snapshot[sq] = convert_piece(native[sq]);
  }

  Piece king_piece = player == virgo::WHITE ? W_KING : B_KING;
  int king_square = -1;
  for (int sq = 0; sq < 64; ++sq) {
    if (snapshot[sq] == king_piece) {
      king_square = sq;
      break;
    }
  }

  if (king_square == -1) {
    throw std::runtime_error("King not found on board");
  }

  return SquareAttacked(snapshot, king_square, enemy_is_white);
}

bool BitboardState::is_checkmate() const {
  auto legal_moves = generate_legal_moves();
  return legal_moves.empty() && is_check();
}

bool BitboardState::is_stalemate() const {
  auto legal_moves = generate_legal_moves();
  return legal_moves.empty() && !is_check();
}

Piece BitboardState::get_piece_at(int square) const {
  return convert_piece(board_[square]);
}

bool BitboardState::is_capture(const Move &move) const {
  uint16_t encoded = move.encoded ? move.encoded : ensure_encoded(move);
  auto type = static_cast<virgo::MoveType>(MOVE_TYPE(encoded));
  return IsCaptureType(type);
}

uint16_t BitboardState::ensure_encoded(const Move &move) const {
  std::vector<Move> legal_moves = generate_legal_moves();
  for (const Move &candidate : legal_moves) {
    if (candidate.from == move.from && candidate.to == move.to) {
      if (move.promotion == EMPTY || candidate.promotion == move.promotion) {
        return candidate.encoded;
      }
    }
  }
  throw std::runtime_error("Requested move is not legal in current position");
}

Move BitboardState::decode_move(uint16_t encoded) const {
  virgo::Player mover = white_to_move() ? virgo::WHITE : virgo::BLACK;
  return decode_move(encoded, mover);
}

Move BitboardState::decode_move(uint16_t encoded, virgo::Player mover) const {
  int from = MOVE_FROM(encoded);
  int to = MOVE_TO(encoded);
  auto type = static_cast<virgo::MoveType>(MOVE_TYPE(encoded));
  Piece promotion = PromotionFromType(type, mover);
  return Move(from, to, promotion, encoded);
}

std::string BitboardState::move_to_uci(uint16_t encoded) const {
  int from = MOVE_FROM(encoded);
  int to = MOVE_TO(encoded);
  std::string uci;
  uci.reserve(5);
  uci.push_back(static_cast<char>('a' + (from % 8)));
  uci.push_back(static_cast<char>('1' + (from / 8)));
  uci.push_back(static_cast<char>('a' + (to % 8)));
  uci.push_back(static_cast<char>('1' + (to / 8)));

  auto type = static_cast<virgo::MoveType>(MOVE_TYPE(encoded));
  Piece promo =
      PromotionFromType(type, white_to_move() ? virgo::WHITE : virgo::BLACK);
  if (promo != EMPTY) {
    char promo_char = 'q';
    int piece_code = promo < 0 ? -promo : promo;
    switch (piece_code) {
    case 2:
      promo_char = 'n';
      break;
    case 3:
      promo_char = 'b';
      break;
    case 4:
      promo_char = 'r';
      break;
    case 5:
    default:
      promo_char = 'q';
      break;
    }
    uci.push_back(promo_char);
  }

  return uci;
}

BitboardState
BitboardState::from_components(const std::array<int8_t, 64> &encoded,
                               bool white_to_move, const std::string &castling,
                               int en_passant_square, int halfmove_clock,
                               int fullmove_number) {
  BitboardState state;
  state.set_from_components(encoded, white_to_move, castling, en_passant_square,
                            halfmove_clock, fullmove_number);
  return state;
}

void BitboardState::set_from_components(const std::array<int8_t, 64> &encoded,
                                        bool white_to_move,
                                        const std::string &castling,
                                        int en_passant_square,
                                        int halfmove_clock,
                                        int fullmove_number) {
  ensure_initialized();
  std::array<int8_t, 64> encoded_copy = encoded;
  uint8_t castling_mask = CastlingMaskFromString(castling);
  unsigned int ep_square = (en_passant_square >= 0 && en_passant_square < 64)
                               ? static_cast<unsigned int>(en_passant_square)
                               : virgo::INVALID;
  uint8_t halfmove = static_cast<uint8_t>(std::clamp(halfmove_clock, 0, 255));

  board_.set_state(encoded_copy, white_to_move ? virgo::WHITE : virgo::BLACK,
                   castling_mask, ep_square, halfmove, 0);

  fullmove_number_ = fullmove_number < 1 ? 1 : fullmove_number;
}

uint64_t BitboardState::occupancy() const {
  return const_cast<virgo::Chessboard &>(board_).occupancy();
}

uint64_t BitboardState::zobrist() const {
  uint64_t key = 0;
  const virgo::Chessboard &native = board_;

  for (int sq = 0; sq < 64; ++sq) {
    Piece piece = convert_piece(native[sq]);
    int idx = piece_index(piece);
    if (idx >= 0) {
      key ^= g_piece_keys[idx][sq];
    }
  }

  if (white_to_move()) {
    key ^= g_side_key;
  }

  uint8_t castle_perm = native.get_castling_permissions() & 0x0F;
  key ^= g_castle_keys[castle_perm];

  unsigned int ep_square = native.get_en_passant_square();
  if (ep_square != virgo::INVALID) {
    key ^= g_en_passant_keys[ep_square % 8];
  }

  return key;
}

Piece BitboardState::convert_piece(
    const std::pair<virgo::Piece, virgo::Player> &cell) const {
  if (cell.first == virgo::EMPTY) {
    return EMPTY;
  }
  bool white = cell.second == virgo::WHITE;
  switch (cell.first) {
  case virgo::PAWN:
    return white ? W_PAWN : B_PAWN;
  case virgo::ROOK:
    return white ? W_ROOK : B_ROOK;
  case virgo::KNIGHT:
    return white ? W_KNIGHT : B_KNIGHT;
  case virgo::BISHOP:
    return white ? W_BISHOP : B_BISHOP;
  case virgo::KING:
    return white ? W_KING : B_KING;
  case virgo::QUEEN:
    return white ? W_QUEEN : B_QUEEN;
  default:
    return EMPTY;
  }
}

char BitboardState::piece_to_char(Piece piece) const {
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

std::string
BitboardState::castling_string(const virgo::Chessboard &board) const {
  virgo::Chessboard copy = board;
  std::string rights;
  if (copy.can_castle_king_side<virgo::WHITE>())
    rights.push_back('K');
  if (copy.can_castle_queen_side<virgo::WHITE>())
    rights.push_back('Q');
  if (copy.can_castle_king_side<virgo::BLACK>())
    rights.push_back('k');
  if (copy.can_castle_queen_side<virgo::BLACK>())
    rights.push_back('q');

  if (rights.empty()) {
    return "-";
  }
  return rights;
}

std::string
BitboardState::en_passant_string(const virgo::Chessboard &board) const {
  unsigned int square = board.get_en_passant_square();
  if (square == virgo::INVALID) {
    return "-";
  }
  char file = static_cast<char>('a' + (square & 7));
  char rank = static_cast<char>('1' + (square >> 3));
  return std::string{file, rank};
}

int BitboardState::parse_fullmove_number(const std::string &fen) const {
  std::istringstream iss(fen);
  std::string board_part, turn, castling, en_passant, halfmove, fullmove;
  if (!(iss >> board_part >> turn >> castling >> en_passant >> halfmove >>
        fullmove)) {
    return 1;
  }
  try {
    int value = std::stoi(fullmove);
    return value < 1 ? 1 : value;
  } catch (...) {
    return 1;
  }
}

} // namespace bitboard
