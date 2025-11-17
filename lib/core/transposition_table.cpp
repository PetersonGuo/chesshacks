#include "core/transposition_table.h"

void TranspositionTable::store(uint64_t key, int depth, int score,
                               TTEntryType type, uint16_t best_move) {
  TableShard &shard = get_shard(key);
  std::lock_guard<std::mutex> lock(shard.mutex);
  // Only store if this is a deeper search or doesn't exist
  auto it = shard.table.find(key);
  if (it == shard.table.end() || it->second.depth <= depth) {
    shard.table[key] = {depth, score, type, best_move};
  }
}

bool TranspositionTable::probe(uint64_t key, int depth, int alpha, int beta,
                               int &score, uint16_t &best_move) {
  const TableShard &shard = get_shard(key);
  std::lock_guard<std::mutex> lock(shard.mutex);
  auto it = shard.table.find(key);
  if (it == shard.table.end()) {
    return false;
  }

  const TTEntry &entry = it->second;

  // Only use if the stored depth is >= current depth
  if (entry.depth < depth) {
    return false;
  }

  // Return best move for move ordering
  best_move = entry.best_move;

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
  for (auto &shard : shards) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.table.clear();
  }
}

size_t TranspositionTable::size() const {
  size_t total = 0;
  for (const auto &shard : shards) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    total += shard.table.size();
  }
  return total;
}

uint16_t TranspositionTable::get_best_move(uint64_t key) const {
  const TableShard &shard = get_shard(key);
  std::lock_guard<std::mutex> lock(shard.mutex);
  auto it = shard.table.find(key);
  if (it == shard.table.end()) {
    return 0;
  }
  return it->second.best_move;
}
