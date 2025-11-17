#pragma once

#include "bitboard/bitboard_state.h"
#include "move_ordering.h"
#include "transposition_table.h"

#include <functional>

namespace search_internal {

struct EvalProvider {
  std::function<int(const bitboard::BitboardState &)> board_eval;

  int operator()(const bitboard::BitboardState &board) const {
    return board_eval(board);
  }
};

EvalProvider
MakeEvalProvider(const std::function<int(const bitboard::BitboardState &)> &);

int enforce_depth_limit(int depth);

int quiescence_search(bitboard::BitboardState &board, int alpha, int beta,
                      bool maximizingPlayer, const EvalProvider &evaluate,
                      int q_depth = 0, int max_q_depth = 2);

int alpha_beta_internal(bitboard::BitboardState &board, int depth, int alpha,
                        int beta, bool maximizingPlayer,
                        const EvalProvider &evaluate, TranspositionTable &tt,
                        bool use_quiescence = true,
                        KillerMoves *killers = nullptr, int ply = 0,
                        HistoryTable *history = nullptr,
                        bool allow_null_move = false,
                        CounterMoveTable *counters = nullptr,
                        ContinuationHistory *cont_history = nullptr,
                        int prev_piece = 0, int prev_to = -1);

int alpha_beta_impl(bitboard::BitboardState board, int depth, int alpha,
                    int beta, bool maximizingPlayer,
                    const EvalProvider &evaluate, TranspositionTable *tt,
                    int num_threads, KillerMoves *killers,
                    HistoryTable *history, CounterMoveTable *counters);

} // namespace search_internal
