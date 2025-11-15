#include "chess_board.h"
#include <algorithm>
#include <cctype>
#include <sstream>

ChessBoard::ChessBoard() {
  board.fill(EMPTY);
  white_to_move = true;
  castling_rights = 0;
  en_passant_square = -1;
  halfmove_clock = 0;
  fullmove_number = 1;
}

ChessBoard::ChessBoard(const std::string &fen) : ChessBoard() { from_fen(fen); }

void ChessBoard::from_fen(const std::string &fen) {
  board.fill(EMPTY);
  std::istringstream iss(fen);
  std::string board_str, turn, castling, ep, halfmove, fullmove;

  iss >> board_str >> turn >> castling >> ep >> halfmove >> fullmove;

  // Parse board
  int rank = 7, file = 0;
  for (char c : board_str) {
    if (c == '/') {
      rank--;
      file = 0;
    } else if (isdigit(c)) {
      file += (c - '0');
    } else {
      Piece piece = EMPTY;
      switch (tolower(c)) {
      case 'p':
        piece = isupper(c) ? W_PAWN : B_PAWN;
        break;
      case 'n':
        piece = isupper(c) ? W_KNIGHT : B_KNIGHT;
        break;
      case 'b':
        piece = isupper(c) ? W_BISHOP : B_BISHOP;
        break;
      case 'r':
        piece = isupper(c) ? W_ROOK : B_ROOK;
        break;
      case 'q':
        piece = isupper(c) ? W_QUEEN : B_QUEEN;
        break;
      case 'k':
        piece = isupper(c) ? W_KING : B_KING;
        break;
      }
      board[rank * 8 + file] = piece;
      file++;
    }
  }

  // Parse turn
  white_to_move = (turn == "w");

  // Parse castling rights
  castling_rights = 0;
  if (castling.find('K') != std::string::npos)
    castling_rights |= 0x1;
  if (castling.find('Q') != std::string::npos)
    castling_rights |= 0x2;
  if (castling.find('k') != std::string::npos)
    castling_rights |= 0x4;
  if (castling.find('q') != std::string::npos)
    castling_rights |= 0x8;

  // Parse en passant
  if (ep != "-") {
    int ep_file = ep[0] - 'a';
    int ep_rank = ep[1] - '1';
    en_passant_square = ep_rank * 8 + ep_file;
  } else {
    en_passant_square = -1;
  }

  halfmove_clock = std::stoi(halfmove);
  fullmove_number = std::stoi(fullmove);
}

std::string ChessBoard::to_fen() const {
  std::ostringstream oss;

  // Board
  for (int rank = 7; rank >= 0; rank--) {
    int empty_count = 0;
    for (int file = 0; file < 8; file++) {
      Piece p = board[rank * 8 + file];
      if (p == EMPTY) {
        empty_count++;
      } else {
        if (empty_count > 0) {
          oss << empty_count;
          empty_count = 0;
        }
        char c;
        switch (abs(p)) {
        case 1:
          c = 'p';
          break;
        case 2:
          c = 'n';
          break;
        case 3:
          c = 'b';
          break;
        case 4:
          c = 'r';
          break;
        case 5:
          c = 'q';
          break;
        case 6:
          c = 'k';
          break;
        default:
          c = '?';
        }
        if (p > 0)
          c = toupper(c);
        oss << c;
      }
    }
    if (empty_count > 0)
      oss << empty_count;
    if (rank > 0)
      oss << '/';
  }

  // Turn
  oss << (white_to_move ? " w " : " b ");

  // Castling
  std::string castling;
  if (castling_rights & 0x1)
    castling += 'K';
  if (castling_rights & 0x2)
    castling += 'Q';
  if (castling_rights & 0x4)
    castling += 'k';
  if (castling_rights & 0x8)
    castling += 'q';
  oss << (castling.empty() ? "-" : castling) << " ";

  // En passant
  if (en_passant_square >= 0) {
    oss << (char)('a' + en_passant_square % 8)
        << (char)('1' + en_passant_square / 8);
  } else {
    oss << "-";
  }

  oss << " " << halfmove_clock << " " << fullmove_number;

  return oss.str();
}

bool ChessBoard::is_valid_square(int rank, int file) const {
  return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

int ChessBoard::to_square(int rank, int file) const { return rank * 8 + file; }

int ChessBoard::find_king(bool white) const {
  Piece king = white ? W_KING : B_KING;
  for (int i = 0; i < 64; i++) {
    if (board[i] == king)
      return i;
  }
  return -1;
}

bool ChessBoard::is_square_attacked(int square, bool by_white) const {
  int rank = square / 8;
  int file = square % 8;

  // Check for pawn attacks
  int pawn_dir = by_white ? 1 : -1;
  Piece enemy_pawn = by_white ? W_PAWN : B_PAWN;
  if (is_valid_square(rank - pawn_dir, file - 1) &&
      board[to_square(rank - pawn_dir, file - 1)] == enemy_pawn)
    return true;
  if (is_valid_square(rank - pawn_dir, file + 1) &&
      board[to_square(rank - pawn_dir, file + 1)] == enemy_pawn)
    return true;

  // Check for knight attacks
  const int knight_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                  {1, -2},  {1, 2},  {2, -1},  {2, 1}};
  Piece enemy_knight = by_white ? W_KNIGHT : B_KNIGHT;
  for (auto &km : knight_moves) {
    int nr = rank + km[0], nf = file + km[1];
    if (is_valid_square(nr, nf) && board[to_square(nr, nf)] == enemy_knight)
      return true;
  }

  // Check for sliding piece attacks (bishop, rook, queen)
  const std::vector<std::pair<int, int>> bishop_dirs = {
      {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const std::vector<std::pair<int, int>> rook_dirs = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  Piece enemy_bishop = by_white ? W_BISHOP : B_BISHOP;
  Piece enemy_rook = by_white ? W_ROOK : B_ROOK;
  Piece enemy_queen = by_white ? W_QUEEN : B_QUEEN;

  for (auto &dir : bishop_dirs) {
    for (int dist = 1; dist < 8; dist++) {
      int nr = rank + dir.first * dist, nf = file + dir.second * dist;
      if (!is_valid_square(nr, nf))
        break;
      Piece p = board[to_square(nr, nf)];
      if (p == enemy_bishop || p == enemy_queen)
        return true;
      if (p != EMPTY)
        break;
    }
  }

  for (auto &dir : rook_dirs) {
    for (int dist = 1; dist < 8; dist++) {
      int nr = rank + dir.first * dist, nf = file + dir.second * dist;
      if (!is_valid_square(nr, nf))
        break;
      Piece p = board[to_square(nr, nf)];
      if (p == enemy_rook || p == enemy_queen)
        return true;
      if (p != EMPTY)
        break;
    }
  }

  // Check for king attacks
  Piece enemy_king = by_white ? W_KING : B_KING;
  for (int dr = -1; dr <= 1; dr++) {
    for (int df = -1; df <= 1; df++) {
      if (dr == 0 && df == 0)
        continue;
      int nr = rank + dr, nf = file + df;
      if (is_valid_square(nr, nf) && board[to_square(nr, nf)] == enemy_king)
        return true;
    }
  }

  return false;
}

bool ChessBoard::is_check() const {
  int king_square = find_king(white_to_move);
  return king_square >= 0 && is_square_attacked(king_square, !white_to_move);
}

void ChessBoard::generate_sliding_moves(
    int square, const std::vector<std::pair<int, int>> &directions,
    std::vector<Move> &moves) const {
  int rank = square / 8, file = square % 8;
  bool is_white = board[square] > 0;

  for (auto &dir : directions) {
    for (int dist = 1; dist < 8; dist++) {
      int nr = rank + dir.first * dist, nf = file + dir.second * dist;
      if (!is_valid_square(nr, nf))
        break;

      int target = to_square(nr, nf);
      Piece target_piece = board[target];

      if (target_piece == EMPTY) {
        moves.push_back(Move(square, target));
      } else if ((is_white && target_piece < 0) ||
                 (!is_white && target_piece > 0)) {
        moves.push_back(Move(square, target));
        break;
      } else {
        break;
      }
    }
  }
}

void ChessBoard::generate_pawn_moves(int square,
                                     std::vector<Move> &moves) const {
  int rank = square / 8, file = square % 8;
  bool is_white = board[square] > 0;
  int dir = is_white ? 1 : -1;
  int start_rank = is_white ? 1 : 6;
  int promote_rank = is_white ? 7 : 0;

  // Forward move
  int forward = to_square(rank + dir, file);
  if (is_valid_square(rank + dir, file) && board[forward] == EMPTY) {
    if (rank + dir == promote_rank) {
      // Promotions
      Piece queen = is_white ? W_QUEEN : B_QUEEN;
      Piece rook = is_white ? W_ROOK : B_ROOK;
      Piece bishop = is_white ? W_BISHOP : B_BISHOP;
      Piece knight = is_white ? W_KNIGHT : B_KNIGHT;
      moves.push_back(Move(square, forward, queen));
      moves.push_back(Move(square, forward, rook));
      moves.push_back(Move(square, forward, bishop));
      moves.push_back(Move(square, forward, knight));
    } else {
      moves.push_back(Move(square, forward));

      // Double move from start
      if (rank == start_rank) {
        int double_forward = to_square(rank + 2 * dir, file);
        if (board[double_forward] == EMPTY) {
          moves.push_back(Move(square, double_forward));
        }
      }
    }
  }

  // Captures
  for (int df : {-1, 1}) {
    if (is_valid_square(rank + dir, file + df)) {
      int capture = to_square(rank + dir, file + df);
      Piece target = board[capture];
      bool can_capture = (is_white && target < 0) || (!is_white && target > 0);

      if (can_capture || capture == en_passant_square) {
        if (rank + dir == promote_rank) {
          // Capture promotions
          Piece queen = is_white ? W_QUEEN : B_QUEEN;
          Piece rook = is_white ? W_ROOK : B_ROOK;
          Piece bishop = is_white ? W_BISHOP : B_BISHOP;
          Piece knight = is_white ? W_KNIGHT : B_KNIGHT;
          moves.push_back(Move(square, capture, queen));
          moves.push_back(Move(square, capture, rook));
          moves.push_back(Move(square, capture, bishop));
          moves.push_back(Move(square, capture, knight));
        } else {
          moves.push_back(Move(square, capture));
        }
      }
    }
  }
}

void ChessBoard::generate_knight_moves(int square,
                                       std::vector<Move> &moves) const {
  int rank = square / 8, file = square % 8;
  bool is_white = board[square] > 0;

  const int knight_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                  {1, -2},  {1, 2},  {2, -1},  {2, 1}};
  for (auto &km : knight_moves) {
    int nr = rank + km[0], nf = file + km[1];
    if (!is_valid_square(nr, nf))
      continue;

    int target = to_square(nr, nf);
    Piece target_piece = board[target];

    if (target_piece == EMPTY || (is_white && target_piece < 0) ||
        (!is_white && target_piece > 0)) {
      moves.push_back(Move(square, target));
    }
  }
}

void ChessBoard::generate_bishop_moves(int square,
                                       std::vector<Move> &moves) const {
  const std::vector<std::pair<int, int>> directions = {
      {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  generate_sliding_moves(square, directions, moves);
}

void ChessBoard::generate_rook_moves(int square,
                                     std::vector<Move> &moves) const {
  const std::vector<std::pair<int, int>> directions = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  generate_sliding_moves(square, directions, moves);
}

void ChessBoard::generate_queen_moves(int square,
                                      std::vector<Move> &moves) const {
  const std::vector<std::pair<int, int>> directions = {
      {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  generate_sliding_moves(square, directions, moves);
}

void ChessBoard::generate_king_moves(int square,
                                     std::vector<Move> &moves) const {
  int rank = square / 8, file = square % 8;
  bool is_white = board[square] > 0;

  // Normal moves
  for (int dr = -1; dr <= 1; dr++) {
    for (int df = -1; df <= 1; df++) {
      if (dr == 0 && df == 0)
        continue;
      int nr = rank + dr, nf = file + df;
      if (!is_valid_square(nr, nf))
        continue;

      int target = to_square(nr, nf);
      Piece target_piece = board[target];

      if (target_piece == EMPTY || (is_white && target_piece < 0) ||
          (!is_white && target_piece > 0)) {
        moves.push_back(Move(square, target));
      }
    }
  }

  // Castling
  if (is_white) {
    // Kingside
    if ((castling_rights & 0x1) && board[61] == EMPTY && board[62] == EMPTY &&
        !is_square_attacked(60, false) && !is_square_attacked(61, false) &&
        !is_square_attacked(62, false)) {
      moves.push_back(Move(60, 62));
    }
    // Queenside
    if ((castling_rights & 0x2) && board[59] == EMPTY && board[58] == EMPTY &&
        board[57] == EMPTY && !is_square_attacked(60, false) &&
        !is_square_attacked(59, false) && !is_square_attacked(58, false)) {
      moves.push_back(Move(60, 58));
    }
  } else {
    // Kingside
    if ((castling_rights & 0x4) && board[5] == EMPTY && board[6] == EMPTY &&
        !is_square_attacked(4, true) && !is_square_attacked(5, true) &&
        !is_square_attacked(6, true)) {
      moves.push_back(Move(4, 6));
    }
    // Queenside
    if ((castling_rights & 0x8) && board[3] == EMPTY && board[2] == EMPTY &&
        board[1] == EMPTY && !is_square_attacked(4, true) &&
        !is_square_attacked(3, true) && !is_square_attacked(2, true)) {
      moves.push_back(Move(4, 2));
    }
  }
}

std::vector<Move> ChessBoard::generate_legal_moves() const {
  std::vector<Move> pseudo_legal_moves;

  for (int square = 0; square < 64; square++) {
    Piece piece = board[square];
    if (piece == EMPTY)
      continue;
    if ((white_to_move && piece < 0) || (!white_to_move && piece > 0))
      continue;

    int abs_piece = abs(piece);
    switch (abs_piece) {
    case 1:
      generate_pawn_moves(square, pseudo_legal_moves);
      break;
    case 2:
      generate_knight_moves(square, pseudo_legal_moves);
      break;
    case 3:
      generate_bishop_moves(square, pseudo_legal_moves);
      break;
    case 4:
      generate_rook_moves(square, pseudo_legal_moves);
      break;
    case 5:
      generate_queen_moves(square, pseudo_legal_moves);
      break;
    case 6:
      generate_king_moves(square, pseudo_legal_moves);
      break;
    }
  }

  // Filter out moves that leave king in check
  std::vector<Move> legal_moves;
  for (const Move &move : pseudo_legal_moves) {
    ChessBoard temp = *this;
    temp.make_move(move);
    temp.white_to_move =
        !temp.white_to_move; // Switch back to check if our king is attacked
    if (!temp.is_check()) {
      legal_moves.push_back(move);
    }
  }

  return legal_moves;
}

void ChessBoard::make_move(const Move &move) {
  // Save state for unmake
  BoardState state;
  state.captured_piece = board[move.to];
  state.castling_rights = castling_rights;
  state.en_passant_square = en_passant_square;
  state.halfmove_clock = halfmove_clock;
  state_stack.push_back(state);

  Piece moving_piece = board[move.from];
  board[move.to] = (move.promotion != EMPTY) ? move.promotion : moving_piece;
  board[move.from] = EMPTY;

  // Handle en passant capture
  if (abs(moving_piece) == 1 && move.to == en_passant_square) {
    int capture_square = move.to + (white_to_move ? -8 : 8);
    board[capture_square] = EMPTY;
  }

  // Set en passant square
  en_passant_square = -1;
  if (abs(moving_piece) == 1 && abs(move.to - move.from) == 16) {
    en_passant_square = (move.from + move.to) / 2;
  }

  // Handle castling
  if (abs(moving_piece) == 6 && abs(move.to - move.from) == 2) {
    if (move.to > move.from) {
      // Kingside
      int rook_from = move.from + 3;
      int rook_to = move.from + 1;
      board[rook_to] = board[rook_from];
      board[rook_from] = EMPTY;
    } else {
      // Queenside
      int rook_from = move.from - 4;
      int rook_to = move.from - 1;
      board[rook_to] = board[rook_from];
      board[rook_from] = EMPTY;
    }
  }

  // Update castling rights
  if (moving_piece == W_KING)
    castling_rights &= ~0x3;
  if (moving_piece == B_KING)
    castling_rights &= ~0xC;
  if (move.from == 0 || move.to == 0)
    castling_rights &= ~0x8;
  if (move.from == 7 || move.to == 7)
    castling_rights &= ~0x4;
  if (move.from == 56 || move.to == 56)
    castling_rights &= ~0x2;
  if (move.from == 63 || move.to == 63)
    castling_rights &= ~0x1;

  // Update clocks
  if (abs(moving_piece) == 1 || state.captured_piece != EMPTY) {
    halfmove_clock = 0;
  } else {
    halfmove_clock++;
  }

  if (!white_to_move)
    fullmove_number++;
  white_to_move = !white_to_move;
}

void ChessBoard::unmake_move(const Move &move) {
  if (state_stack.empty())
    return;

  BoardState state = state_stack.back();
  state_stack.pop_back();

  white_to_move = !white_to_move;

  Piece moving_piece = board[move.to];
  if (move.promotion != EMPTY) {
    moving_piece = white_to_move ? W_PAWN : B_PAWN;
  }

  board[move.from] = moving_piece;
  board[move.to] = state.captured_piece;

  // Undo en passant capture
  if (abs(moving_piece) == 1 && move.to == state.en_passant_square &&
      state.captured_piece == EMPTY) {
    int capture_square = move.to + (white_to_move ? -8 : 8);
    board[capture_square] = white_to_move ? B_PAWN : W_PAWN;
  }

  // Undo castling
  if (abs(moving_piece) == 6 && abs(move.to - move.from) == 2) {
    if (move.to > move.from) {
      int rook_to = move.from + 1;
      int rook_from = move.from + 3;
      board[rook_from] = board[rook_to];
      board[rook_to] = EMPTY;
    } else {
      int rook_to = move.from - 1;
      int rook_from = move.from - 4;
      board[rook_from] = board[rook_to];
      board[rook_to] = EMPTY;
    }
  }

  castling_rights = state.castling_rights;
  en_passant_square = state.en_passant_square;
  halfmove_clock = state.halfmove_clock;

  if (white_to_move)
    fullmove_number--;
}

bool ChessBoard::is_checkmate() const {
  return is_check() && generate_legal_moves().empty();
}

bool ChessBoard::is_stalemate() const {
  return !is_check() && generate_legal_moves().empty();
}
