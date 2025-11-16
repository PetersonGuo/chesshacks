#include "transposition_table.h"

void TranspositionTable::store(const std::string &fen, int depth, int score,
                               TTEntryType type,
                               const std::string &best_move_fen) {
  std::lock_guard<std::mutex> lock(mutex);
  // Only store if this is a deeper search or doesn't exist
  auto it = table.find(fen);
  if (it == table.end() || it->second.depth <= depth) {
    table[fen] = {depth, score, type, best_move_fen};
  }
}

bool TranspositionTable::probe(const std::string &fen, int depth, int alpha,
                               int beta, int &score,
                               std::string &best_move_fen) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = table.find(fen);
  if (it == table.end()) {
    return false;
  }

  const TTEntry &entry = it->second;

  // Only use if the stored depth is >= current depth
  if (entry.depth < depth) {
    return false;
  }

  // Return best move for move ordering
  best_move_fen = entry.best_move_fen;

  // Check if we can use this cached value
  if (entry.type == EXACT) {
    score = entry.score;
    return true;
  } else if (entry.type == LOWER_BOUND && entry.score >= beta) {
    score = entry.score;
    return true;
  } else if (entry.type == UPPER_BOUND && entry.score <= alpha) {
    score = entry.score;
    return true;
  }

  return false;
}

void TranspositionTable::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  table.clear();
}

size_t TranspositionTable::size() const {
  std::lock_guard<std::mutex> lock(mutex);
  return table.size();
}

std::string TranspositionTable::get_best_move(const std::string &fen) const {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = table.find(fen);
  if (it == table.end()) {
    return "";
  }
  return it->second.best_move_fen;
}
