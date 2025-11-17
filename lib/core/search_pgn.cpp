#include "core/utils.h"

#include <cctype>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using bitboard::BitboardState;

bool parse_and_apply_san(BitboardState &board, const std::string &san_move) {
  std::string move = san_move;
  if (!move.empty() && (move.back() == '+' || move.back() == '#')) {
    move.pop_back();
  }

  if (move == "O-O" || move == "0-0" || move == "o-o") {
    bool is_white = board.white_to_move();
    int king_rank = is_white ? 0 : 7;
    int king_from = king_rank * 8 + 4;
    int king_to = king_rank * 8 + 6;
    board.make_move(BoardMove(king_from, king_to));
    return true;
  } else if (move == "O-O-O" || move == "0-0-0" || move == "o-o-o") {
    bool is_white = board.white_to_move();
    int king_rank = is_white ? 0 : 7;
    int king_from = king_rank * 8 + 4;
    int king_to = king_rank * 8 + 2;
    board.make_move(BoardMove(king_from, king_to));
    return true;
  }

  BoardPiece piece_type = EMPTY;
  int piece_idx = 0;
  if (move.length() > 0 && std::isupper(move[0])) {
    char p = std::toupper(move[0]);
    bool is_white = board.white_to_move();
    switch (p) {
    case 'K':
      piece_type = is_white ? W_KING : B_KING;
      break;
    case 'Q':
      piece_type = is_white ? W_QUEEN : B_QUEEN;
      break;
    case 'R':
      piece_type = is_white ? W_ROOK : B_ROOK;
      break;
    case 'B':
      piece_type = is_white ? W_BISHOP : B_BISHOP;
      break;
    case 'N':
      piece_type = is_white ? W_KNIGHT : B_KNIGHT;
      break;
    default:
      return false;
    }
    piece_idx = 1;
  } else {
    bool is_white = board.white_to_move();
    piece_type = is_white ? W_PAWN : B_PAWN;
  }

  if (piece_idx < static_cast<int>(move.length()) && move[piece_idx] == 'x') {
    piece_idx++;
  }

  BoardPiece promotion = EMPTY;
  size_t eq_pos = move.find('=');
  if (eq_pos != std::string::npos && eq_pos + 1 < move.length()) {
    char prom_char = std::toupper(move[eq_pos + 1]);
    bool is_white = board.white_to_move();
    switch (prom_char) {
    case 'Q':
      promotion = is_white ? W_QUEEN : B_QUEEN;
      break;
    case 'R':
      promotion = is_white ? W_ROOK : B_ROOK;
      break;
    case 'B':
      promotion = is_white ? W_BISHOP : B_BISHOP;
      break;
    case 'N':
      promotion = is_white ? W_KNIGHT : B_KNIGHT;
      break;
    default:
      return false;
    }
    move = move.substr(0, eq_pos);
  }

  if (move.length() < 2)
    return false;

  std::string target_str = move.substr(move.length() - 2);
  if (target_str[0] < 'a' || target_str[0] > 'h' || target_str[1] < '1' ||
      target_str[1] > '8') {
    return false;
  }

  int target_file = target_str[0] - 'a';
  int target_rank = target_str[1] - '1';
  int target_square = target_rank * 8 + target_file;

  std::vector<BoardMove> legal_moves = board.generate_legal_moves();

  std::vector<BoardMove> candidates;
  for (const BoardMove &m : legal_moves) {
    if (m.to == target_square) {
      BoardPiece moved_piece = board.get_piece_at(m.from);
      if (std::abs(moved_piece) == std::abs(piece_type) ||
          (piece_type == (board.white_to_move() ? W_PAWN : B_PAWN) &&
           std::abs(moved_piece) == std::abs(W_PAWN))) {
        if (promotion == EMPTY || m.promotion == promotion) {
          candidates.push_back(m);
        }
      }
    }
  }

  if (candidates.size() > 1 &&
      piece_idx < static_cast<int>(move.length() - 2)) {
    std::string disambig =
        move.substr(piece_idx, move.length() - piece_idx - 2);
    std::vector<BoardMove> filtered;
    for (const BoardMove &m : candidates) {
      bool matches = true;
      for (char c : disambig) {
        if (std::isdigit(c)) {
          int rank = c - '1';
          if (m.from / 8 != rank) {
            matches = false;
            break;
          }
        } else if (std::isalpha(c)) {
          int file = std::tolower(c) - 'a';
          if (m.from % 8 != file) {
            matches = false;
            break;
          }
        }
      }
      if (matches) {
        filtered.push_back(m);
      }
    }
    if (!filtered.empty()) {
      candidates = filtered;
    }
  }

  if (!candidates.empty()) {
    board.make_move(candidates[0]);
    return true;
  }

  return false;
}

std::string pgn_to_fen(const std::string &pgn) {
  BitboardState board(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  std::string moves_text = pgn;

  std::istringstream iss(moves_text);
  std::string line;
  std::string clean_moves;
  bool in_headers = true;

  while (std::getline(iss, line)) {
    if (line.empty()) {
      in_headers = false;
      continue;
    }
    if (in_headers && (line[0] == '[' || line.find("[Event") == 0 ||
                       line.find("[Site") == 0 || line.find("[Date") == 0 ||
                       line.find("[White") == 0 || line.find("[Black") == 0 ||
                       line.find("[Result") == 0)) {
      continue;
    }
    if (!in_headers || line[0] != '[') {
      clean_moves += line + " ";
      in_headers = false;
    }
  }

  std::regex move_pattern(R"(\d+\.\s*([^\s]+\s+[^\s]+|[^\s]+))");
  std::smatch matches;
  std::string::const_iterator search_start(clean_moves.cbegin());

  std::vector<std::string> parsed_moves;
  while (std::regex_search(search_start, clean_moves.cend(), matches,
                           move_pattern)) {
    std::string move_pair = matches[1].str();

    std::istringstream move_stream(move_pair);
    std::string white_move, black_move;
    move_stream >> white_move;
    if (move_stream >> black_move) {
      parsed_moves.push_back(white_move);
      parsed_moves.push_back(black_move);
    } else {
      parsed_moves.push_back(white_move);
    }

    search_start = matches[0].second;
  }

  if (parsed_moves.empty()) {
    std::istringstream move_stream(clean_moves);
    std::string move;
    while (move_stream >> move) {
      if (move.find('.') == std::string::npos && move != "1-0" &&
          move != "0-1" && move != "1/2-1/2" && move != "*" &&
          move.find('-') == std::string::npos) {
        parsed_moves.push_back(move);
      }
    }
  }

  for (const std::string &move : parsed_moves) {
    if (!parse_and_apply_san(board, move)) {
      continue;
    }
  }

  return board.to_fen();
}

BitboardState pgn_to_bitboard(const std::string &pgn) {
  BitboardState board(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  std::string moves_text = pgn;
  std::istringstream iss(moves_text);
  std::string line;
  std::string clean_moves;
  bool in_headers = true;

  while (std::getline(iss, line)) {
    if (line.empty()) {
      in_headers = false;
      continue;
    }
    if (in_headers && (line[0] == '[' || line.find("[Event") == 0 ||
                       line.find("[Site") == 0 || line.find("[Date") == 0 ||
                       line.find("[White") == 0 || line.find("[Black") == 0 ||
                       line.find("[Result") == 0)) {
      continue;
    }
    if (!in_headers || line[0] != '[') {
      clean_moves += line + " ";
      in_headers = false;
    }
  }

  std::regex move_pattern(R"(\d+\.\s*([^\s]+\s+[^\s]+|[^\s]+))");
  std::smatch matches;
  std::string::const_iterator search_start(clean_moves.cbegin());

  std::vector<std::string> parsed_moves;
  while (std::regex_search(search_start, clean_moves.cend(), matches,
                           move_pattern)) {
    std::string move_pair = matches[1].str();

    std::istringstream move_stream(move_pair);
    std::string white_move, black_move;
    move_stream >> white_move;
    if (move_stream >> black_move) {
      parsed_moves.push_back(white_move);
      parsed_moves.push_back(black_move);
    } else {
      parsed_moves.push_back(white_move);
    }

    search_start = matches[0].second;
  }

  if (parsed_moves.empty()) {
    std::istringstream move_stream(clean_moves);
    std::string move;
    while (move_stream >> move) {
      if (move.find('.') == std::string::npos && move != "1-0" &&
          move != "0-1" && move != "1/2-1/2" && move != "*" &&
          move.find('-') == std::string::npos) {
        parsed_moves.push_back(move);
      }
    }
  }

  for (const std::string &move : parsed_moves) {
    if (!parse_and_apply_san(board, move)) {
      continue;
    }
  }

  return board;
}
