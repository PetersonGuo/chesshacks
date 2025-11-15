#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <climits>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

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
};

// Thread-safe transposition table class
class TranspositionTable {
public:
  void store(const std::string &fen, int depth, int score, TTEntryType type);
  bool probe(const std::string &fen, int depth, int alpha, int beta,
             int &score);
  void clear();
  size_t size() const;

private:
  std::unordered_map<std::string, TTEntry> table;
  mutable std::mutex mutex;
};

int alpha_beta(const std::string &fen, int depth, int alpha, int beta,
               bool maximizingPlayer,
               const std::function<int(const std::string &)> &evaluate);

// Version with transposition table
int alpha_beta_with_tt(const std::string &fen, int depth, int alpha, int beta,
                       bool maximizingPlayer,
                       const std::function<int(const std::string &)> &evaluate,
                       TranspositionTable &tt);

// Parallel version - evaluates root moves in parallel
int alpha_beta_parallel(const std::string &fen, int depth, int alpha, int beta,
                        bool maximizingPlayer,
                        const std::function<int(const std::string &)> &evaluate,
                        int num_threads = 0); // 0 = auto-detect

// Parallel with transposition table
int alpha_beta_parallel_with_tt(const std::string &fen, int depth, int alpha, int beta,
                                bool maximizingPlayer,
                                const std::function<int(const std::string &)> &evaluate,
                                TranspositionTable &tt,
                                int num_threads = 0);

#endif // FUNCTIONS_H
