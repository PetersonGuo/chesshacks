#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <climits>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "chess_board.h"

const int MIN = INT_MIN;
const int MAX = INT_MAX;
const int MAX_DEPTH = 64; // Maximum search depth for killer moves

// Transposition table entry types
enum TTEntryType {
  EXACT,       // Exact score
  LOWER_BOUND, // Alpha cutoff (score >= stored value)
  UPPER_BOUND  // Beta cutoff (score <= stored value)
};

// Killer moves table (stores 2 killer moves per ply)
struct KillerMoves {
  std::string killers[MAX_DEPTH][2];
  
  void store(int ply, const std::string &move_fen);
  bool is_killer(int ply, const std::string &move_fen) const;
  void clear();
};

// History heuristic table (piece-to-square)
class HistoryTable {
public:
  void update(int piece, int to_square, int depth);
  int get_score(int piece, int to_square) const;
  void clear();
  void age(); // Age history scores to favor recent moves
  
private:
  // [piece_type + 6][to_square] - piece_type ranges from -6 to 6
  int history[13][64] = {{0}};
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
  bool probe(const std::string &fen, int depth, int alpha, int beta,
             int &score, std::string &best_move_fen);
  void clear();
  size_t size() const;

private:
  std::unordered_map<std::string, TTEntry> table;
  mutable std::mutex mutex;
};

// 1. BASIC: Bare-bones alpha-beta (no optimizations) - BACKUP
int alpha_beta_basic(const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer,
                     const std::function<int(const std::string &)> &evaluate);

// 2. OPTIMIZED: Full optimizations (TT + move ordering + parallel)
int alpha_beta_optimized(const std::string &fen, int depth, int alpha, int beta,
                         bool maximizingPlayer,
                         const std::function<int(const std::string &)> &evaluate,
                         TranspositionTable *tt = nullptr,
                         int num_threads = 0,
                         KillerMoves *killers = nullptr,
                         HistoryTable *history = nullptr);

// 3. CUDA: GPU-accelerated search (falls back to optimized if CUDA unavailable)
int alpha_beta_cuda(const std::string &fen, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable *tt = nullptr,
                    KillerMoves *killers = nullptr,
                    HistoryTable *history = nullptr);

// HELPER FUNCTIONS
int get_piece_square_value(Piece piece, int square);
int evaluate_with_pst(const std::string &fen);
bool is_cuda_available();
std::string get_cuda_info();

#endif // FUNCTIONS_H
