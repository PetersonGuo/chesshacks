#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <mutex>
#include <string>
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
  std::string best_move_fen; // Store best move for move ordering
};

// Thread-safe transposition table class
class TranspositionTable {
public:
  void store(const std::string &fen, int depth, int score, TTEntryType type,
             const std::string &best_move_fen = "");
  bool probe(const std::string &fen, int depth, int alpha, int beta, int &score,
             std::string &best_move_fen);
  void clear();
  size_t size() const;
  std::string get_best_move(const std::string &fen) const;

private:
  std::unordered_map<std::string, TTEntry> table;
  mutable std::mutex mutex;
};

#endif // TRANSPOSITION_TABLE_H
