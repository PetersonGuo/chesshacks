#ifndef SEARCH_H
#define SEARCH_H

#include "chess_board.h"
#include "move_ordering.h"
#include "transposition_table.h"
#include <climits>
#include <functional>
#include <string>
#include <vector>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

// Multi-PV search result
struct PVLine {
  std::string uci_move; // Best move in UCI format
  int score;            // Evaluation score
  int depth;            // Search depth
  std::string pv;       // Principal variation (sequence of moves)
};

// 1. BASIC: Bare-bones alpha-beta (no optimizations) - BACKUP
int alpha_beta_basic(const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer,
                     const std::function<int(const std::string &)> &evaluate);

// 2. OPTIMIZED: Full optimizations (TT + move ordering + parallel)
int alpha_beta_optimized(
    const std::string &fen, int depth, int alpha, int beta,
    bool maximizingPlayer,
    const std::function<int(const std::string &)> &evaluate,
    TranspositionTable *tt = nullptr, int num_threads = 0,
    KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
    CounterMoveTable *counters = nullptr);

// 3. CUDA: GPU-accelerated search (falls back to optimized if CUDA unavailable)
int alpha_beta_cuda(const std::string &fen, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable *tt = nullptr,
                    KillerMoves *killers = nullptr,
                    HistoryTable *history = nullptr,
                    CounterMoveTable *counters = nullptr);

// Quiescence search to avoid horizon effect
int quiescence_search(ChessBoard &board, int alpha, int beta,
                      bool maximizingPlayer,
                      const std::function<int(const std::string &)> &evaluate,
                      int q_depth = 0, int max_q_depth = 4);

// Iterative deepening wrapper with aspiration windows
int iterative_deepening(const std::string &fen, int max_depth, int alpha,
                        int beta, bool maximizingPlayer,
                        const std::function<int(const std::string &)> &evaluate,
                        TranspositionTable &tt, KillerMoves *killers = nullptr,
                        HistoryTable *history = nullptr);

// Find best move (returns FEN after move)
std::string
find_best_move(const std::string &fen, int depth,
               const std::function<int(const std::string &)> &evaluate,
               TranspositionTable *tt = nullptr, int num_threads = 0,
               KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
               CounterMoveTable *counters = nullptr);

// Get best move in UCI format
std::string
get_best_move_uci(const std::string &fen, int depth,
                  const std::function<int(const std::string &)> &evaluate,
                  TranspositionTable *tt = nullptr, int num_threads = 0,
                  KillerMoves *killers = nullptr,
                  HistoryTable *history = nullptr,
                  CounterMoveTable *counters = nullptr);

// Search for multiple principal variations
std::vector<PVLine>
multi_pv_search(const std::string &fen, int depth, int num_lines,
                const std::function<int(const std::string &)> &evaluate,
                TranspositionTable *tt = nullptr, int num_threads = 0,
                KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
                CounterMoveTable *counters = nullptr);

#endif // SEARCH_H
