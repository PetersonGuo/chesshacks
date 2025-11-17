#ifndef SEARCH_H
#define SEARCH_H

#include "bitboard/bitboard_state.h"
#include "move_ordering.h"
#include "transposition_table.h"
#include <climits>
#include <functional>
#include <string>
#include <vector>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

using BoardEvalFn = std::function<int(const bitboard::BitboardState &)>;

// Multi-PV search result
struct PVLine {
  std::string uci_move; // Best move in UCI format
  int score;            // Evaluation score
  int depth;            // Search depth
  std::string pv;       // Principal variation (sequence of moves)
};

// Alpha-beta search with all optimizations (TT + move ordering + parallel)
int alpha_beta(bitboard::BitboardState board, int depth, int alpha, int beta,
               bool maximizingPlayer, const BoardEvalFn &evaluate,
               TranspositionTable *tt = nullptr, int num_threads = 0,
               KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
               CounterMoveTable *counters = nullptr);

int alpha_beta_builtin(bitboard::BitboardState board, int depth, int alpha,
                       int beta, bool maximizingPlayer,
                       TranspositionTable *tt = nullptr, int num_threads = 0,
                       KillerMoves *killers = nullptr,
                       HistoryTable *history = nullptr,
                       CounterMoveTable *counters = nullptr);

#ifdef CUDA_ENABLED
// CUDA: GPU-accelerated search (falls back to CPU alpha-beta if unavailable)
int alpha_beta_cuda(bitboard::BitboardState board, int depth, int alpha,
                    int beta, bool maximizingPlayer,
                    const BoardEvalFn &evaluate,
                    TranspositionTable *tt = nullptr,
                    KillerMoves *killers = nullptr,
                    HistoryTable *history = nullptr,
                    CounterMoveTable *counters = nullptr);
#endif

// Find best move (returns resulting board state)
bitboard::BitboardState find_best_move(bitboard::BitboardState board, int depth,
                                       const BoardEvalFn &evaluate,
                                       TranspositionTable *tt = nullptr,
                                       int num_threads = 0,
                                       KillerMoves *killers = nullptr,
                                       HistoryTable *history = nullptr,
                                       CounterMoveTable *counters = nullptr);

bitboard::BitboardState find_best_move_builtin(
    bitboard::BitboardState board, int depth, TranspositionTable *tt = nullptr,
    int num_threads = 0, KillerMoves *killers = nullptr,
    HistoryTable *history = nullptr, CounterMoveTable *counters = nullptr);

// Get best move in UCI format
std::string get_best_move_uci(bitboard::BitboardState board, int depth,
                              const BoardEvalFn &evaluate,
                              TranspositionTable *tt = nullptr,
                              int num_threads = 0,
                              KillerMoves *killers = nullptr,
                              HistoryTable *history = nullptr,
                              CounterMoveTable *counters = nullptr);

std::string get_best_move_uci_builtin(bitboard::BitboardState board, int depth,
                                      TranspositionTable *tt = nullptr,
                                      int num_threads = 0,
                                      KillerMoves *killers = nullptr,
                                      HistoryTable *history = nullptr,
                                      CounterMoveTable *counters = nullptr);

// Search for multiple principal variations
std::vector<PVLine> multi_pv_search(bitboard::BitboardState board, int depth,
                                    int num_lines, const BoardEvalFn &evaluate,
                                    TranspositionTable *tt = nullptr,
                                    int num_threads = 0,
                                    KillerMoves *killers = nullptr,
                                    HistoryTable *history = nullptr,
                                    CounterMoveTable *counters = nullptr);

void set_max_search_depth(int depth);

#endif // SEARCH_H
