#include "core/search_internal.h"

#include "core/evaluation.h"
#include "core/search.h"

#include <algorithm>
#include <atomic>
#include <vector>

using bitboard::BitboardState;

namespace {
constexpr int kDefaultMaxDepth = 5;
std::atomic<int> g_max_search_depth{kDefaultMaxDepth};
} // namespace

void set_max_search_depth(int depth) {
  if (depth <= 0)
    depth = kDefaultMaxDepth;
  g_max_search_depth.store(depth, std::memory_order_relaxed);
}

namespace search_internal {

int enforce_depth_limit(int depth) {
  int limit = g_max_search_depth.load(std::memory_order_relaxed);
  if (limit > 0 && depth > limit)
    return limit;
  return depth;
}

EvalProvider
MakeEvalProvider(const std::function<int(const BitboardState &)> &eval_func) {
  EvalProvider provider;
  if (eval_func) {
    provider.board_eval = eval_func;
  } else {
    provider.board_eval = [](const BitboardState &board) {
      return evaluate(board);
    };
  }
  return provider;
}

int quiescence_search(BitboardState &board, int alpha, int beta,
                      bool maximizingPlayer, const EvalProvider &evaluate,
                      int q_depth, int max_q_depth) {
  if (q_depth >= max_q_depth) {
    return evaluate(board);
  }

  int stand_pat = evaluate(board);

  if (maximizingPlayer) {
    if (stand_pat >= beta) {
      return beta;
    }
    if (alpha < stand_pat) {
      alpha = stand_pat;
    }
  } else {
    if (stand_pat <= alpha) {
      return alpha;
    }
    if (beta > stand_pat) {
      beta = stand_pat;
    }
  }

  std::vector<BoardMove> all_moves = board.generate_legal_moves();
  std::vector<BoardMove> captures;
  for (const BoardMove &move : all_moves) {
    if (board.is_capture(move) || move.promotion != EMPTY) {
      captures.push_back(move);
    }
  }

  std::sort(captures.begin(), captures.end(),
            [&board](const BoardMove &a, const BoardMove &b) {
              return mvv_lva_score(board, a) > mvv_lva_score(board, b);
            });

  if (maximizingPlayer) {
    int maxEval = stand_pat;
    for (const BoardMove &move : captures) {
      board.make_move(move);
      int eval = quiescence_search(board, alpha, beta, false, evaluate,
                                   q_depth + 1, max_q_depth);
      board.unmake_move(move);

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;
      }
    }
    return maxEval;
  } else {
    int minEval = stand_pat;
    for (const BoardMove &move : captures) {
      board.make_move(move);
      int eval = quiescence_search(board, alpha, beta, true, evaluate,
                                   q_depth + 1, max_q_depth);
      board.unmake_move(move);
      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        break;
      }
    }
    return minEval;
  }
}

} // namespace search_internal
