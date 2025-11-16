#include "opening_book.h"
#include "chess_board.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <random>

// Polyglot Zobrist hash implementation with standard keys
uint64_t OpeningBook::polyglot_hash(const std::string &fen) {
  // Standard Polyglot Zobrist random numbers (fixed seed from spec)
  // These are the official Polyglot keys - 781 random 64-bit numbers
  static std::vector<uint64_t> random_values;

  if (random_values.empty()) {
    // Initialize with Polyglot's specific PRNG (seed = 0)
    std::mt19937_64 rng(0);
    random_values.resize(781);
    for (int i = 0; i < 781; i++) {
      random_values[i] = rng();
    }
  }

  uint64_t hash = 0;
  ChessBoard board;
  board.from_fen(fen);

  // Hash pieces (indices 0-767)
  // Polyglot order: white pieces then black pieces
  // For each piece type: pawn, knight, bishop, rook, queen, king
  for (int sq = 0; sq < 64; sq++) {
    Piece p = board.get_piece_at(sq);
    if (p != EMPTY) {
      int piece_type, color;
      if (p >= 6) { // Black pieces
        color = 1;
        piece_type = p - 6;
      } else { // White pieces
        color = 0;
        piece_type = p;
      }

      // Polyglot uses files from left to right, ranks from bottom to top
      // Convert our square (0=a1, 63=h8) to Polyglot format
      int file = sq % 8;
      int rank = sq / 8;
      int polyglot_sq = rank * 8 + file;

      // Index: color * 384 + piece_type * 64 + square
      int index = color * 384 + piece_type * 64 + polyglot_sq;
      hash ^= random_values[index];
    }
  }

  // Hash castling rights (indices 768-771)
  std::string castling = "";
  size_t space_count = 0;
  for (char c : fen) {
    if (c == ' ')
      space_count++;
    if (space_count == 2) {
      size_t next_space = fen.find(' ', fen.find(c) + 1);
      castling = fen.substr(fen.find(c) + 1, next_space - fen.find(c) - 1);
      break;
    }
  }
  if (castling.find('K') != std::string::npos)
    hash ^= random_values[768];
  if (castling.find('Q') != std::string::npos)
    hash ^= random_values[769];
  if (castling.find('k') != std::string::npos)
    hash ^= random_values[770];
  if (castling.find('q') != std::string::npos)
    hash ^= random_values[771];

  // Hash en passant (indices 772-779)
  std::string ep = "";
  space_count = 0;
  for (char c : fen) {
    if (c == ' ')
      space_count++;
    if (space_count == 3) {
      size_t next_space = fen.find(' ', fen.find(c) + 1);
      if (next_space == std::string::npos)
        next_space = fen.length();
      ep = fen.substr(fen.find(c) + 1, next_space - fen.find(c) - 1);
      break;
    }
  }
  if (ep != "-" && ep.length() == 2) {
    int file = ep[0] - 'a';
    hash ^= random_values[772 + file];
  }

  // Hash side to move (index 780)
  if (fen.find(" b ") != std::string::npos) {
    hash ^= random_values[780];
  }

  return hash;
}

std::string OpeningBook::decode_move(uint16_t move) {
  // Polyglot move format:
  // bits 0-5: to square
  // bits 6-11: from square
  // bits 12-14: promotion (0=none, 1=N, 2=B, 3=R, 4=Q)

  int to_sq = move & 0x3F;
  int from_sq = (move >> 6) & 0x3F;
  int promo = (move >> 12) & 0x7;

  // Convert to UCI
  char from_file = 'a' + (from_sq % 8);
  char from_rank = '1' + (from_sq / 8);
  char to_file = 'a' + (to_sq % 8);
  char to_rank = '1' + (to_sq / 8);

  std::string uci;
  uci += from_file;
  uci += from_rank;
  uci += to_file;
  uci += to_rank;

  // Add promotion
  if (promo > 0) {
    const char promo_chars[] = {' ', 'n', 'b', 'r', 'q'};
    if (promo < 5) {
      uci += promo_chars[promo];
    }
  }

  return uci;
}

bool OpeningBook::load(const std::string &book_path) {
  std::ifstream file(book_path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  entries.clear();

  while (file.good()) {
    BookEntry entry;
    char buffer[16];

    file.read(buffer, 16);
    if (file.gcount() < 16)
      break;

    // Parse big-endian Polyglot format
    entry.key = 0;
    for (int i = 0; i < 8; i++) {
      entry.key = (entry.key << 8) | (unsigned char)buffer[i];
    }

    entry.move = ((unsigned char)buffer[8] << 8) | (unsigned char)buffer[9];
    entry.weight = ((unsigned char)buffer[10] << 8) | (unsigned char)buffer[11];
    entry.learn = ((unsigned char)buffer[12] << 24) |
                  ((unsigned char)buffer[13] << 16) |
                  ((unsigned char)buffer[14] << 8) | (unsigned char)buffer[15];

    entries.push_back(entry);
  }

  file.close();

  // Sort by key for binary search
  std::sort(
      entries.begin(), entries.end(),
      [](const BookEntry &a, const BookEntry &b) { return a.key < b.key; });

  loaded = !entries.empty();
  return loaded;
}

std::vector<BookMove> OpeningBook::probe(const std::string &fen) {
  std::vector<BookMove> moves;

  if (!loaded)
    return moves;

  uint64_t hash = polyglot_hash(fen);

  // Binary search for matching positions
  auto lower = std::lower_bound(
      entries.begin(), entries.end(), hash,
      [](const BookEntry &e, uint64_t h) { return e.key < h; });

  // Collect all moves for this position
  while (lower != entries.end() && lower->key == hash) {
    BookMove bm;
    bm.uci_move = decode_move(lower->move);
    bm.weight = lower->weight;
    moves.push_back(bm);
    ++lower;
  }

  return moves;
}

std::string OpeningBook::probe_best(const std::string &fen) {
  auto moves = probe(fen);
  if (moves.empty())
    return "";

  // Return highest weighted move
  auto best = std::max_element(
      moves.begin(), moves.end(),
      [](const BookMove &a, const BookMove &b) { return a.weight < b.weight; });
  return best->uci_move;
}

std::string OpeningBook::probe_weighted(const std::string &fen) {
  auto moves = probe(fen);
  if (moves.empty())
    return "";

  // Weighted random selection
  int total_weight = 0;
  for (const auto &m : moves) {
    total_weight += m.weight;
  }

  if (total_weight == 0) {
    return moves[0].uci_move;
  }

  int rand_val = rand() % total_weight;
  int current = 0;

  for (const auto &m : moves) {
    current += m.weight;
    if (rand_val < current) {
      return m.uci_move;
    }
  }

  return moves[0].uci_move;
}

void OpeningBook::clear() {
  entries.clear();
  loaded = false;
}
