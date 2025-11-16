#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>

// Transposition table entry types
enum TTEntryType {
  EXACT,       // Exact score
  LOWER_BOUND, // Alpha cutoff (score >= stored value)
  UPPER_BOUND  // Beta cutoff (score <= stored value)
};

struct TTEntry {
  int depth;
  int score;
  TTEntryType type;
  uint16_t best_move = 0; // Encoded move for ordering
};

// Thread-safe transposition table class
class TranspositionTable {
public:
  void store(uint64_t key, int depth, int score, TTEntryType type,
             uint16_t best_move = 0);
  bool probe(uint64_t key, int depth, int alpha, int beta, int &score,
             uint16_t &best_move);
  void clear();
  size_t size() const;
  uint16_t get_best_move(uint64_t key) const;

private:
  static constexpr size_t kShardCount = 64;
  static_assert((kShardCount & (kShardCount - 1)) == 0,
                "kShardCount must be power of two");

  struct TableShard {
    std::unordered_map<uint64_t, TTEntry> table;
    mutable std::mutex mutex;
  };

  std::array<TableShard, kShardCount> shards;

  inline size_t shard_index(uint64_t key) const {
    return key & (kShardCount - 1);
  }

  TableShard &get_shard(uint64_t key) {
    return shards[shard_index(key)];
  }

  const TableShard &get_shard(uint64_t key) const {
    return shards[shard_index(key)];
  }
};

#endif // TRANSPOSITION_TABLE_H
