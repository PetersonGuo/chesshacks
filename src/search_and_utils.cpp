#include "bitboard/bitboard_state.h"
#include "evaluation.h"
#include "search.h"
#include "utils.h"
#ifdef CUDA_ENABLED
#include "cuda/cuda_utils.h"
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <functional>
#include <future>
#include <iostream>
#include <nanobind/nanobind.h>
#include <random>
#include <regex>
#include <sstream>
#include <thread>
#include <vector>

namespace nb = nanobind;
using bitboard::BitboardState;

namespace {
constexpr int kDefaultMaxDepth = 5;
std::atomic<int> g_max_search_depth{kDefaultMaxDepth};

inline int enforce_depth_limit(int depth) {
  int limit = g_max_search_depth.load(std::memory_order_relaxed);
  if (limit > 0 && depth > limit)
    return limit;
  return depth;
}
} // namespace

void set_max_search_depth(int depth) {
  if (depth <= 0)
    depth = kDefaultMaxDepth;
  g_max_search_depth.store(depth, std::memory_order_relaxed);
}

struct EvalProvider {
  std::function<int(const std::string &)> fen_eval;
  std::function<int(const BitboardState &)> board_eval;
  bool use_python = true;

  int operator()(const std::string &fen) const {
    if (use_python) {
      nb::gil_scoped_acquire gil;
      return fen_eval(fen);
    }
    return fen_eval(fen);
  }

  int operator()(const BitboardState &board) const {
    if (board_eval) {
      return board_eval(board);
    }
    if (use_python) {
      nb::gil_scoped_acquire gil;
      return fen_eval(board.to_fen());
    }
    return fen_eval(board.to_fen());
  }
};

static EvalProvider
MakeEvalProvider(const std::function<int(const std::string &)> &evaluate,
                 bool default_python) {
  EvalProvider provider;
  provider.fen_eval = evaluate;
  provider.use_python = default_python;

  using FenEvalPtr = int (*)(const std::string &);
  if (auto fn_ptr = evaluate.template target<FenEvalPtr>()) {
    if (*fn_ptr == static_cast<FenEvalPtr>(&evaluate_with_pst)) {
      provider.board_eval = [](const BitboardState &board) {
        return evaluate_with_pst(board);
      };
      provider.use_python = false;
    } else if (*fn_ptr == static_cast<FenEvalPtr>(&evaluate_material)) {
      provider.board_eval = [](const BitboardState &board) {
        return evaluate_material(board);
      };
      provider.use_python = false;
    }
  }

  return provider;
}

// ============================================================================
// QUIESCENCE SEARCH
// ============================================================================

// Quiescence search to avoid horizon effect (only search captures)
int quiescence_search(BitboardState &board, int alpha, int beta,
                      bool maximizingPlayer, const EvalProvider &evaluate,
                      int q_depth = 0, int max_q_depth = 2) {
  // Limit quiescence depth to prevent infinite recursion
  if (q_depth >= max_q_depth) {
    return evaluate(board);
  }

  // Stand-pat evaluation
  int stand_pat = evaluate(board);

  if (maximizingPlayer) {
    if (stand_pat >= beta) {
      return beta; // Beta cutoff
    }
    if (alpha < stand_pat) {
      alpha = stand_pat;
    }
  } else {
    if (stand_pat <= alpha) {
      return alpha; // Alpha cutoff
    }
    if (beta > stand_pat) {
      beta = stand_pat;
    }
  }

  // Generate only capture moves
  std::vector<Move> all_moves = board.generate_legal_moves();
  std::vector<Move> captures;
  for (const Move &move : all_moves) {
    if (board.is_capture(move) || move.promotion != EMPTY) {
      captures.push_back(move);
    }
  }

  // Order captures by MVV-LVA
  std::sort(captures.begin(), captures.end(),
            [&board](const Move &a, const Move &b) {
              return mvv_lva_score(board, a) > mvv_lva_score(board, b);
            });

  if (maximizingPlayer) {
    int maxEval = stand_pat;
    for (const Move &move : captures) {
      board.make_move(move);
      int eval = quiescence_search(board, alpha, beta, false, evaluate,
                                   q_depth + 1, max_q_depth);
      board.unmake_move(move);

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break; // Beta cutoff
      }
    }
    return maxEval;
  } else {
    int minEval = stand_pat;
    for (const Move &move : captures) {
      board.make_move(move);
      int eval = quiescence_search(board, alpha, beta, true, evaluate,
                                   q_depth + 1, max_q_depth);
      board.unmake_move(move);
      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        break; // Alpha cutoff
      }
    }
    return minEval;
  }
}

// ============================================================================
// 1. BASIC VERSION - Bare bones alpha-beta pruning, no optimizations
// ============================================================================
static int alpha_beta_basic_board(BitboardState &board, int depth, int alpha,
                                  int beta, bool maximizingPlayer,
                                  const EvalProvider &evaluate) {
  if (depth == 0) {
    return evaluate(board);
  }

  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    if (board.is_check()) {
      return maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      return 0;
    }
  }

  if (maximizingPlayer) {
    int maxEval = MIN;
    for (const Move &move : legal_moves) {
      Piece singular_piece = board.get_piece_at(move.from);
      board.make_move(move);
      int eval = alpha_beta_basic_board(board, depth - 1, alpha, beta, false,
                                        evaluate);
      board.unmake_move(move);

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;
      }
    }
    return maxEval;
  } else {
    int minEval = MAX;
    for (const Move &move : legal_moves) {
      Piece singular_piece = board.get_piece_at(move.from);
      board.make_move(move);
      int eval =
          alpha_beta_basic_board(board, depth - 1, alpha, beta, true, evaluate);
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

static int alpha_beta_basic_internal(BitboardState board, int depth, int alpha,
                                     int beta, bool maximizingPlayer,
                                     const EvalProvider &evaluate) {
  return alpha_beta_basic_board(board, depth, alpha, beta, maximizingPlayer,
                                evaluate);
}

static int alpha_beta_basic_internal(const std::string &fen, int depth,
                                     int alpha, int beta, bool maximizingPlayer,
                                     const EvalProvider &evaluate) {
  BitboardState board(fen);
  return alpha_beta_basic_internal(board, depth, alpha, beta, maximizingPlayer,
                                   evaluate);
}

int alpha_beta_basic(const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer,
                     const std::function<int(const std::string &)> &evaluate) {
  depth = enforce_depth_limit(depth);
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  return alpha_beta_basic_internal(fen, depth, alpha, beta, maximizingPlayer,
                                   eval_provider);
}

int alpha_beta_basic(BitboardState board, int depth, int alpha, int beta,
                     bool maximizingPlayer,
                     const std::function<int(const std::string &)> &evaluate) {
  depth = enforce_depth_limit(depth);
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  return alpha_beta_basic_internal(board, depth, alpha, beta, maximizingPlayer,
                                   eval_provider);
}

int alpha_beta_basic_builtin(const std::string &fen, int depth, int alpha,
                             int beta, bool maximizingPlayer) {
  depth = enforce_depth_limit(depth);
  std::function<int(const std::string &)> material_eval_fn =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(material_eval_fn, false);
  return alpha_beta_basic_internal(fen, depth, alpha, beta, maximizingPlayer,
                                   eval_provider);
}

int alpha_beta_basic_builtin(BitboardState board, int depth, int alpha,
                             int beta, bool maximizingPlayer) {
  depth = enforce_depth_limit(depth);
  std::function<int(const std::string &)> material_eval_fn =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(material_eval_fn, false);
  return alpha_beta_basic_internal(board, depth, alpha, beta, maximizingPlayer,
                                   eval_provider);
}

// ============================================================================
// 2. OPTIMIZED VERSION - With TT, move ordering, quiescence, and optional
// multithreading
// ============================================================================

// Internal recursive function with all optimizations
// ============================================================================
// OPTIMIZED ALPHA-BETA WITH ALL FEATURES
// ============================================================================

static int alpha_beta_internal(
    BitboardState &board, int depth, int alpha, int beta, bool maximizingPlayer,
    const EvalProvider &evaluate, TranspositionTable &tt,
    bool use_quiescence = true, KillerMoves *killers = nullptr, int ply = 0,
    HistoryTable *history = nullptr, bool allow_null_move = false,
    CounterMoveTable *counters = nullptr,
    ContinuationHistory *cont_history = nullptr, int prev_piece = 0,
    int prev_to = -1) {
  const uint64_t board_key = board.zobrist();
  int cached_score;
  uint16_t tt_best_move = 0;
  if (tt.probe(board_key, depth, alpha, beta, cached_score, tt_best_move)) {
    return cached_score;
  }

  // Razoring - prune at low depths when evaluation is far below alpha
  if (!board.is_check() && depth <= 3 && depth > 0) {
    int static_eval = evaluate(board);
    int razor_margin = 300 * depth; // More aggressive at lower depths

    if (static_eval + razor_margin < alpha) {
      // Position is so bad that even with margin, unlikely to raise alpha
      // Do a quiescence search to verify
      int q_score =
          quiescence_search(board, alpha, beta, maximizingPlayer, evaluate);

      if (q_score + razor_margin < alpha) {
        return q_score; // Confirmed bad position, return quiescence score
      }
    }
  }

  // At depth 0, enter quiescence search to avoid horizon effect
  if (depth == 0) {
    if (use_quiescence) {
      int q_score =
          quiescence_search(board, alpha, beta, maximizingPlayer, evaluate);
      tt.store(board_key, 0, q_score, EXACT, 0);
      return q_score;
    } else {
      int score = evaluate(board);
      tt.store(board_key, 0, score, EXACT, 0);
      return score;
    }
  }

  std::vector<Move> legal_moves = board.generate_legal_moves();

  // Terminal position (checkmate or stalemate)
  if (legal_moves.empty()) {
    int score;
    if (board.is_check()) {
      score = maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      score = 0; // Stalemate
    }
    tt.store(board_key, depth, score, EXACT, 0);
    return score;
  }

  // Internal Iterative Deepening (IID)
  // If we don't have a TT move at high depths, do a shallow search to find one
  if (tt_best_move == 0 && depth >= 5) {
    // Do a reduced depth search to populate TT with a best move
    int iid_depth = depth - 2;
    BitboardState iid_board = board;
    alpha_beta_internal(iid_board, iid_depth, alpha, beta, maximizingPlayer,
                        evaluate, tt, use_quiescence, killers, ply, history,
                        allow_null_move, counters, cont_history, prev_piece,
                        prev_to);
    // After this search, the TT should have a best move for this position
    tt.probe(board_key, iid_depth, alpha, beta, cached_score, tt_best_move);
  }

  // Singular Extensions
  // If we have a TT move and it appears significantly better than alternatives,
  // extend the search depth for this node
  int extension = 0;
  if (tt_best_move != 0 && depth >= 6) {
    // Perform a reduced search excluding the TT move
    // to see if any other move comes close
    int singular_beta = cached_score - depth * 2; // Margin for singularity
    int singular_depth = depth / 2;

    // Search all moves except the TT move with reduced depth
    int best_alternative = MIN;
    for (const Move &move : legal_moves) {
      if (move.encoded == tt_best_move)
        continue;
      Piece singular_piece = board.get_piece_at(move.from);
      BitboardState alt_board = board;
      alt_board.make_move(move);
      int score = alpha_beta_internal(
          alt_board, singular_depth, singular_beta - 1, singular_beta,
          !maximizingPlayer, evaluate, tt, use_quiescence, killers, ply + 1,
          history, true, counters, cont_history, singular_piece, move.to);

      best_alternative = std::max(best_alternative, score);

      // Early exit if we find a move that's close to the TT move
      if (best_alternative >= singular_beta) {
        break;
      }
    }

    // If no alternative came close, extend the search
    if (best_alternative < singular_beta) {
      extension = 1;
    }
  }

  // Move ordering with TT, killers, MVV-LVA, history, promotions
  order_moves(board, legal_moves, tt_best_move, killers, ply, history, counters,
              prev_piece, prev_to, cont_history);

  int original_alpha = alpha;
  uint16_t best_move_code = 0;
  int moves_searched = 0;

  if (maximizingPlayer) {
    int maxEval = MIN;

    // Futility pruning setup - only at low depths
    bool use_futility = false;
    int futility_margin = 0;
    int static_eval = 0;

    if (depth <= 6 && !board.is_check()) {
      static_eval = evaluate(board);
      futility_margin = 100 + 50 * depth; // Depth 1: 150, Depth 6: 400
      use_futility = true;
    }

    for (const Move &move : legal_moves) {
      Piece moving_piece = board.get_piece_at(move.from);
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);
      bool is_quiet = !is_capture && !is_promotion;
      board.make_move(move);
      int eval;

      // Futility pruning - skip quiet moves that can't raise alpha
      if (use_futility && is_quiet && moves_searched > 0 && !board.is_check()) {
        if (static_eval + futility_margin < alpha) {
          board.unmake_move(move);
          moves_searched++;
          continue; // Skip this move
        }
      }

      auto recurse = [&](int new_depth, int new_alpha, int new_beta,
                         bool new_maximizing,
                         bool new_allow_null_move = true) -> int {
        return alpha_beta_internal(
            board, new_depth, new_alpha, new_beta, new_maximizing, evaluate, tt,
            use_quiescence, killers, ply + 1, history, new_allow_null_move,
            counters, cont_history, moving_piece, move.to);
      };

      // Principal Variation Search (PVS)
      if (moves_searched == 0) {
        eval = recurse(depth - 1 + extension, alpha, beta, false);
      } else {
        bool do_lmr = false;
        int reduction = 0;

        if (depth >= 3 && moves_searched >= 4 && is_quiet &&
            !board.is_check()) {
          do_lmr = true;
          reduction = (moves_searched >= 8) ? 2 : 1;
        }

        if (do_lmr) {
          eval = recurse(depth - 1 - reduction, alpha, alpha + 1, false);
          if (eval > alpha && eval < beta) {
            eval = recurse(depth - 1, alpha, beta, false);
          } else if (eval > alpha) {
            eval = recurse(depth - 1, alpha, alpha + 1, false);
            if (eval > alpha && eval < beta) {
              eval = recurse(depth - 1, alpha, beta, false);
            }
          }
        } else {
          eval = recurse(depth - 1, alpha, alpha + 1, false);
          if (eval > alpha && eval < beta) {
            eval = recurse(depth - 1, alpha, beta, false);
          }
        }
      }

      board.unmake_move(move);
      moves_searched++;

      if (eval >= beta) {
        if (killers && is_quiet) {
          killers->store(ply, move.encoded);
        }
        if (history && is_quiet) {
          history->update(moving_piece, move.to, depth);
        }
        if (counters && prev_to >= 0) {
          counters->store(prev_piece, prev_to, move.encoded);
        }
        if (cont_history && prev_to >= 0 && is_quiet) {
          cont_history->update(prev_piece, prev_to, moving_piece, move.to,
                               depth);
        }
        tt.store(board_key, depth, beta, LOWER_BOUND, move.encoded);
        return beta;
      }

      if (eval > maxEval) {
        maxEval = eval;
        best_move_code = move.encoded;
        if (cont_history && prev_to >= 0 && is_quiet) {
          cont_history->update(prev_piece, prev_to, moving_piece, move.to,
                               depth);
        }
      }

      alpha = std::max(alpha, eval);
    }

    // Store in transposition table with best move
    if (maxEval <= original_alpha) {
      tt.store(board_key, depth, maxEval, UPPER_BOUND, best_move_code);
    } else if (maxEval >= beta) {
      tt.store(board_key, depth, maxEval, LOWER_BOUND, best_move_code);
    } else {
      tt.store(board_key, depth, maxEval, EXACT, best_move_code);
    }

    return maxEval;
  } else {
    int minEval = MAX;

    bool use_futility = false;
    int futility_margin = 0;
    int static_eval = 0;

    if (depth <= 6 && !board.is_check()) {
      static_eval = evaluate(board);
      futility_margin = 100 + 50 * depth;
      use_futility = true;
    }

    for (const Move &move : legal_moves) {
      Piece moving_piece = board.get_piece_at(move.from);
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);
      bool is_quiet = !is_capture && !is_promotion;
      board.make_move(move);

      int eval;

      if (use_futility && is_quiet && moves_searched > 0 && !board.is_check()) {
        if (static_eval - futility_margin > beta) {
          board.unmake_move(move);
          moves_searched++;
          continue;
        }
      }

      auto recurse = [&](int new_depth, int new_alpha, int new_beta,
                         bool new_maximizing,
                         bool new_allow_null_move = true) -> int {
        return alpha_beta_internal(
            board, new_depth, new_alpha, new_beta, new_maximizing, evaluate, tt,
            use_quiescence, killers, ply + 1, history, new_allow_null_move,
            counters, cont_history, moving_piece, move.to);
      };

      if (moves_searched == 0) {
        eval = recurse(depth - 1 + extension, alpha, beta, true);
      } else {
        bool do_lmr = false;
        int reduction = 0;

        if (depth >= 3 && moves_searched >= 4 && is_quiet &&
            !board.is_check()) {
          do_lmr = true;
          reduction = (moves_searched >= 8) ? 2 : 1;
        }

        if (do_lmr) {
          eval = recurse(depth - 1 - reduction, beta - 1, beta, true);
          if (eval < beta && eval > alpha) {
            eval = recurse(depth - 1, alpha, beta, true);
          } else if (eval < beta) {
            eval = recurse(depth - 1, beta - 1, beta, true);
            if (eval < beta && eval > alpha) {
              eval = recurse(depth - 1, alpha, beta, true);
            }
          }
        } else {
          eval = recurse(depth - 1, beta - 1, beta, true);
          if (eval < beta && eval > alpha) {
            eval = recurse(depth - 1, alpha, beta, true);
          }
        }
      }

      board.unmake_move(move);
      moves_searched++;

      if (eval <= alpha) {
        if (killers && is_quiet) {
          killers->store(ply, move.encoded);
        }
        if (history && is_quiet) {
          history->update(moving_piece, move.to, depth);
        }
        if (counters && prev_to >= 0) {
          counters->store(prev_piece, prev_to, move.encoded);
        }
        if (cont_history && prev_to >= 0 && is_quiet) {
          cont_history->update(prev_piece, prev_to, moving_piece, move.to,
                               depth);
        }
        tt.store(board_key, depth, alpha, UPPER_BOUND, move.encoded);
        return alpha;
      }

      if (eval < minEval) {
        minEval = eval;
        best_move_code = move.encoded;
        if (cont_history && prev_to >= 0 && is_quiet) {
          cont_history->update(prev_piece, prev_to, moving_piece, move.to,
                               depth);
        }
      }

      beta = std::min(beta, eval);
    }

    if (minEval >= beta) {
      tt.store(board_key, depth, minEval, LOWER_BOUND, best_move_code);
    } else if (minEval <= original_alpha) {
      tt.store(board_key, depth, minEval, UPPER_BOUND, best_move_code);
    } else {
      tt.store(board_key, depth, minEval, EXACT, best_move_code);
    }

    return minEval;
  }

  // Should never reach here
  return maximizingPlayer ? alpha : beta;
}

// Iterative deepening wrapper with aspiration windows
int iterative_deepening(BitboardState &root_board, int max_depth, int alpha,
                        int beta, bool maximizingPlayer,
                        const EvalProvider &evaluate, TranspositionTable &tt,
                        KillerMoves *killers, HistoryTable *history,
                        CounterMoveTable *counters,
                        ContinuationHistory *cont_history) {
  max_depth = enforce_depth_limit(max_depth);
  int best_score = 0;
  const int ASPIRATION_WINDOW = 50; // Initial window size

  // Search with increasing depth
  for (int depth = 1; depth <= max_depth; depth++) {
    BitboardState search_board = root_board;
    int current_alpha = alpha;
    int current_beta = beta;

    // Use aspiration windows for depth >= 3
    if (depth >= 3) {
      current_alpha = best_score - ASPIRATION_WINDOW;
      current_beta = best_score + ASPIRATION_WINDOW;

      // Ensure we don't go beyond original bounds
      if (current_alpha < alpha)
        current_alpha = alpha;
      if (current_beta > beta)
        current_beta = beta;
    }

    // Search with aspiration window
    int score =
        alpha_beta_internal(search_board, depth, current_alpha, current_beta,
                            maximizingPlayer, evaluate, tt, true, killers, 0,
                            history, true, counters, cont_history, 0, -1);

    // If score falls outside window, re-search with full window
    if (depth >= 3 && (score <= current_alpha || score >= current_beta)) {
      // Failed low or high - re-search with full window
      BitboardState retry_board = root_board;
      score = alpha_beta_internal(
          retry_board, depth, alpha, beta, maximizingPlayer, evaluate, tt, true,
          killers, 0, history, true, counters, cont_history, 0, -1);
    }

    best_score = score;

    // Early exit on mate found
    if (best_score <= MIN + 1000 || best_score >= MAX - 1000) {
      break;
    }
  }

  return best_score;
}

// 2. OPTIMIZED: Full production version with all optimizations
static int alpha_beta_optimized_impl(
    BitboardState board, int depth, int alpha, int beta, bool maximizingPlayer,
    const EvalProvider &evaluate, TranspositionTable *tt, int num_threads,
    KillerMoves *killers, HistoryTable *history, CounterMoveTable *counters) {
  depth = enforce_depth_limit(depth);
  TranspositionTable local_tt;
  TranspositionTable &tt_ref = tt ? *tt : local_tt;

  // Create local killer/history/counter tables if not provided
  KillerMoves local_killers;
  HistoryTable local_history;
  CounterMoveTable local_counters;
  ContinuationHistory local_cont_history;
  KillerMoves *killers_ptr = killers ? killers : &local_killers;
  HistoryTable *history_ptr = history ? history : &local_history;
  CounterMoveTable *counters_ptr = counters ? counters : &local_counters;
  ContinuationHistory *cont_history_ptr = &local_cont_history;

  const uint64_t root_key = board.zobrist();
  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    if (board.is_check()) {
      return maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      return 0;
    }
  }

  unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
  int default_threads =
      (hw_threads > 1) ? static_cast<int>(std::min(hw_threads, 4u)) : 1;

  bool auto_threads = (num_threads <= 0);
  int effective_threads =
      auto_threads ? default_threads : std::max(1, std::min(num_threads, 64));
  effective_threads =
      std::min(effective_threads, static_cast<int>(legal_moves.size()));
  const int original_alpha = alpha;
  const int original_beta = beta;

  if (effective_threads <= 1 || depth <= 2) {
    return iterative_deepening(board, depth, alpha, beta, maximizingPlayer,
                               evaluate, tt_ref, killers_ptr, history_ptr,
                               counters_ptr, cont_history_ptr);
  }

  // First do iterative deepening up to depth-1 to populate TT
  if (depth > 1) {
    BitboardState prep_board = board;
    iterative_deepening(prep_board, depth - 1, alpha, beta, maximizingPlayer,
                        evaluate, tt_ref, killers_ptr, history_ptr,
                        counters_ptr, cont_history_ptr);
  }

  // Order root moves using TT, killer, history information
  uint16_t root_tt_move = tt_ref.get_best_move(root_key);
  order_moves(board, legal_moves, root_tt_move, killers_ptr, 0, history_ptr,
              counters_ptr, 0, -1, cont_history_ptr);

  struct MoveScore {
    Move move;
    int score;
  };

  std::vector<std::future<MoveScore>> futures;

  // Release GIL only when spawning threads
  nb::gil_scoped_release gil_release;

  auto evaluate_move = [&](const Move &move) -> MoveScore {
    BitboardState local_board = board;
    Piece moving_piece = local_board.get_piece_at(move.from);
    local_board.make_move(move);

    KillerMoves thread_killers = *killers_ptr;
    HistoryTable thread_history = *history_ptr;
    CounterMoveTable thread_counters;
    ContinuationHistory thread_cont_history;

    int score = alpha_beta_internal(
        local_board, depth - 1, alpha, beta, !maximizingPlayer, evaluate,
        tt_ref, true, &thread_killers, 1, &thread_history, true,
        &thread_counters, &thread_cont_history, moving_piece, move.to);

    return {move, score};
  };

  // Launch threads for moves - batch processing for better efficiency
  std::vector<MoveScore> results;
  results.reserve(legal_moves.size());

  for (size_t i = 0; i < legal_moves.size(); i++) {
    // Wait for a slot to become available
    while (futures.size() >= static_cast<size_t>(effective_threads)) {
      // Efficiently wait for any future to complete
      for (auto it = futures.begin(); it != futures.end();) {
        if (it->wait_for(std::chrono::microseconds(100)) ==
            std::future_status::ready) {
          results.push_back(it->get());
          it = futures.erase(it);
        } else {
          ++it;
        }
      }
    }

    futures.push_back(
        std::async(std::launch::async, evaluate_move, legal_moves[i]));
  }

  // Collect remaining results
  for (auto &future : futures) {
    results.push_back(future.get());
  }

  if (results.empty()) {
    return maximizingPlayer ? MIN : MAX;
  }

  auto best_it =
      maximizingPlayer
          ? std::max_element(results.begin(), results.end(),
                             [](const MoveScore &a, const MoveScore &b) {
                               return a.score < b.score;
                             })
          : std::min_element(results.begin(), results.end(),
                             [](const MoveScore &a, const MoveScore &b) {
                               return a.score < b.score;
                             });

  int best_score = best_it->score;
  uint16_t best_move_code = best_it->move.encoded;

  TTEntryType entry_type = EXACT;
  if (best_score <= original_alpha) {
    entry_type = UPPER_BOUND;
  } else if (best_score >= original_beta) {
    entry_type = LOWER_BOUND;
  }

  tt_ref.store(root_key, depth, best_score, entry_type, best_move_code);
  return best_score;
}

static int alpha_beta_optimized_impl(const std::string &fen, int depth,
                                     int alpha, int beta, bool maximizingPlayer,
                                     const EvalProvider &evaluate,
                                     TranspositionTable *tt, int num_threads,
                                     KillerMoves *killers,
                                     HistoryTable *history,
                                     CounterMoveTable *counters) {
  BitboardState board(fen);
  return alpha_beta_optimized_impl(board, depth, alpha, beta, maximizingPlayer,
                                   evaluate, tt, num_threads, killers, history,
                                   counters);
}

int alpha_beta_optimized(
    const std::string &fen, int depth, int alpha, int beta,
    bool maximizingPlayer,
    const std::function<int(const std::string &)> &evaluate,
    TranspositionTable *tt, int num_threads, KillerMoves *killers,
    HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  return alpha_beta_optimized_impl(fen, depth, alpha, beta, maximizingPlayer,
                                   eval_provider, tt, num_threads, killers,
                                   history, counters);
}

int alpha_beta_optimized(
    BitboardState board, int depth, int alpha, int beta, bool maximizingPlayer,
    const std::function<int(const std::string &)> &evaluate,
    TranspositionTable *tt, int num_threads, KillerMoves *killers,
    HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  return alpha_beta_optimized_impl(board, depth, alpha, beta, maximizingPlayer,
                                   eval_provider, tt, num_threads, killers,
                                   history, counters);
}

int alpha_beta_optimized_builtin(const std::string &fen, int depth, int alpha,
                                 int beta, bool maximizingPlayer,
                                 TranspositionTable *tt, int num_threads,
                                 KillerMoves *killers, HistoryTable *history,
                                 CounterMoveTable *counters) {
  std::function<int(const std::string &)> native_eval =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(native_eval, false);
  return alpha_beta_optimized_impl(fen, depth, alpha, beta, maximizingPlayer,
                                   eval_provider, tt, num_threads, killers,
                                   history, counters);
}

int alpha_beta_optimized_builtin(BitboardState board, int depth, int alpha,
                                 int beta, bool maximizingPlayer,
                                 TranspositionTable *tt, int num_threads,
                                 KillerMoves *killers, HistoryTable *history,
                                 CounterMoveTable *counters) {
  std::function<int(const std::string &)> native_eval =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(native_eval, false);
  return alpha_beta_optimized_impl(board, depth, alpha, beta, maximizingPlayer,
                                   eval_provider, tt, num_threads, killers,
                                   history, counters);
}

// ============================================================================
// 3. CUDA VERSION - GPU-accelerated batch evaluation
// ============================================================================
int alpha_beta_cuda(const std::string &fen, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable *tt, KillerMoves *killers,
                    HistoryTable *history, CounterMoveTable *counters) {
  BitboardState board(fen);
  return alpha_beta_cuda(board, depth, alpha, beta, maximizingPlayer, evaluate,
                         tt, killers, history, counters);
}

int alpha_beta_cuda(BitboardState board, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable *tt, KillerMoves *killers,
                    HistoryTable *history, CounterMoveTable *counters) {
  depth = enforce_depth_limit(depth);
#ifdef CUDA_ENABLED
  if (is_cuda_available()) {
    return alpha_beta_optimized(board, depth, alpha, beta, maximizingPlayer,
                                evaluate, tt, 0, killers, history, counters);
  }
#endif
  return alpha_beta_optimized(board, depth, alpha, beta, maximizingPlayer,
                              evaluate, tt, 0, killers, history, counters);
}

// ============================================================================
// BEST MOVE FINDER - Returns the actual best move, not just score
// ============================================================================
static std::string find_best_move_impl(BitboardState board, int depth,
                                       const EvalProvider &evaluate,
                                       TranspositionTable *tt, int num_threads,
                                       KillerMoves *killers,
                                       HistoryTable *history,
                                       CounterMoveTable *counters,
                                       uint16_t *best_move_code_out = nullptr) {
  depth = enforce_depth_limit(depth);
  // Create local instances if not provided
  TranspositionTable local_tt;
  KillerMoves local_killers;
  HistoryTable local_history;
  CounterMoveTable local_counters;

  if (!tt)
    tt = &local_tt;
  if (!killers)
    killers = &local_killers;
  if (!history)
    history = &local_history;
  if (!counters)
    counters = &local_counters;

  const uint64_t root_key = board.zobrist();
  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    return ""; // No legal moves
  }

  bool maximizing = board.white_to_move();

  // Run the search - this populates TT with best move
  BitboardState search_board = board;
  alpha_beta_optimized_impl(search_board, depth, MIN, MAX, maximizing, evaluate,
                            tt, num_threads, killers, history, counters);

  // Get the best move FEN from TT
  uint16_t best_move_code = tt->get_best_move(root_key);

  if (best_move_code != 0) {
    if (best_move_code_out)
      *best_move_code_out = best_move_code;
    Move move = board.decode_move(best_move_code);
    board.make_move(move);
    std::string result = board.to_fen();
    board.unmake_move(move);
    return result;
  }

  int best_score = maximizing ? MIN : MAX;
  uint16_t fallback_best_code = 0;

  if (num_threads > 1 && legal_moves.size() > 1) {
    nb::gil_scoped_release gil_release;

    struct MoveResult {
      uint16_t encoded;
      int score;
    };

    std::vector<std::future<MoveResult>> futures;
    auto evaluate_move = [&](const Move &move) -> MoveResult {
      BitboardState local_board = board;
      Piece moving_piece = local_board.get_piece_at(move.from);
      local_board.make_move(move);

      KillerMoves thread_killers = *killers;
      HistoryTable thread_history = *history;
      CounterMoveTable thread_counters = *counters;
      ContinuationHistory thread_cont_history;

      int score = alpha_beta_internal(
          local_board, depth - 1, MIN, MAX, !maximizing, evaluate, *tt, true,
          &thread_killers, 1, &thread_history, true, &thread_counters,
          &thread_cont_history, moving_piece, move.to);

      return {move.encoded, score};
    };

    int max_parallel =
        (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    if (max_parallel == 0)
      max_parallel = 4;

    std::vector<MoveResult> results;
    results.reserve(legal_moves.size());

    for (size_t i = 0; i < legal_moves.size(); i++) {
      while (futures.size() >= static_cast<size_t>(max_parallel)) {
        for (auto it = futures.begin(); it != futures.end();) {
          if (it->wait_for(std::chrono::microseconds(100)) ==
              std::future_status::ready) {
            results.push_back(it->get());
            it = futures.erase(it);
          } else {
            ++it;
          }
        }
      }
      futures.push_back(
          std::async(std::launch::async, evaluate_move, legal_moves[i]));
    }

    for (auto &future : futures) {
      results.push_back(future.get());
    }

    for (const auto &result : results) {
      if ((maximizing && result.score > best_score) ||
          (!maximizing && result.score < best_score)) {
        best_score = result.score;
        fallback_best_code = result.encoded;
      }
    }
  } else {
    for (const Move &move : legal_moves) {
      BitboardState local_board = board;
      Piece moving_piece = local_board.get_piece_at(move.from);
      local_board.make_move(move);

      KillerMoves local_killers = *killers;
      HistoryTable local_history = *history;
      CounterMoveTable local_counters = *counters;
      ContinuationHistory local_cont_history;

      int score = alpha_beta_internal(
          local_board, depth - 1, MIN, MAX, !maximizing, evaluate, *tt, true,
          &local_killers, 1, &local_history, true, &local_counters,
          &local_cont_history, moving_piece, move.to);

      if ((maximizing && score > best_score) ||
          (!maximizing && score < best_score)) {
        best_score = score;
        fallback_best_code = move.encoded;
      }
    }
  }

  if (best_move_code_out)
    *best_move_code_out = fallback_best_code;

  if (fallback_best_code == 0) {
    return "";
  }

  BitboardState fallback_board = board;
  Move fallback_move = fallback_board.decode_move(fallback_best_code);
  fallback_board.make_move(fallback_move);
  return fallback_board.to_fen();
}

std::string
find_best_move(const std::string &fen, int depth,
               const std::function<int(const std::string &)> &evaluate,
               TranspositionTable *tt, int num_threads, KillerMoves *killers,
               HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  BitboardState board(fen);
  return find_best_move_impl(board, depth, eval_provider, tt, num_threads,
                             killers, history, counters);
}

std::string
find_best_move(bitboard::BitboardState board, int depth,
               const std::function<int(const std::string &)> &evaluate,
               TranspositionTable *tt, int num_threads, KillerMoves *killers,
               HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  return find_best_move_impl(board, depth, eval_provider, tt, num_threads,
                             killers, history, counters);
}

std::string find_best_move_builtin(const std::string &fen, int depth,
                                   TranspositionTable *tt, int num_threads,
                                   KillerMoves *killers, HistoryTable *history,
                                   CounterMoveTable *counters) {
  std::function<int(const std::string &)> native_eval =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(native_eval, false);
  BitboardState board(fen);
  return find_best_move_impl(board, depth, eval_provider, tt, num_threads,
                             killers, history, counters);
}

std::string find_best_move_builtin(bitboard::BitboardState board, int depth,
                                   TranspositionTable *tt, int num_threads,
                                   KillerMoves *killers, HistoryTable *history,
                                   CounterMoveTable *counters) {
  std::function<int(const std::string &)> native_eval =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(native_eval, false);
  return find_best_move_impl(board, depth, eval_provider, tt, num_threads,
                             killers, history, counters);
}

static std::string convert_best_move_to_uci(BitboardState board,
                                            uint16_t best_move_code) {
  if (best_move_code == 0) {
    return "";
  }
  return board.move_to_uci(best_move_code);
}

// Convert best move FEN back to move notation (UCI format: e.g., "e2e4")
std::string
get_best_move_uci(const std::string &fen, int depth,
                  const std::function<int(const std::string &)> &evaluate,
                  TranspositionTable *tt, int num_threads, KillerMoves *killers,
                  HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  uint16_t best_move_code = 0;
  BitboardState board(fen);
  find_best_move_impl(board, depth, eval_provider, tt, num_threads, killers,
                      history, counters, &best_move_code);
  return convert_best_move_to_uci(board, best_move_code);
}

std::string
get_best_move_uci(BitboardState board, int depth,
                  const std::function<int(const std::string &)> &evaluate,
                  TranspositionTable *tt, int num_threads, KillerMoves *killers,
                  HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  uint16_t best_move_code = 0;
  find_best_move_impl(board, depth, eval_provider, tt, num_threads, killers,
                      history, counters, &best_move_code);
  return convert_best_move_to_uci(board, best_move_code);
}

std::string get_best_move_uci_builtin(const std::string &fen, int depth,
                                      TranspositionTable *tt, int num_threads,
                                      KillerMoves *killers,
                                      HistoryTable *history,
                                      CounterMoveTable *counters) {
  uint16_t best_move_code = 0;
  BitboardState board(fen);
  std::function<int(const std::string &)> material_eval_fn =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(material_eval_fn, false);
  find_best_move_impl(board, depth, eval_provider, tt, num_threads, killers,
                      history, counters, &best_move_code);
  return convert_best_move_to_uci(board, best_move_code);
}

std::string get_best_move_uci_builtin(BitboardState board, int depth,
                                      TranspositionTable *tt, int num_threads,
                                      KillerMoves *killers,
                                      HistoryTable *history,
                                      CounterMoveTable *counters) {
  uint16_t best_move_code = 0;
  std::function<int(const std::string &)> material_eval_fn =
      static_cast<int (*)(const std::string &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(material_eval_fn, false);
  find_best_move_impl(board, depth, eval_provider, tt, num_threads, killers,
                      history, counters, &best_move_code);
  return convert_best_move_to_uci(board, best_move_code);
}

// ============================================================================
// 4. PGN TO FEN - Convert PGN string to FEN string
// ============================================================================

// Helper function to parse SAN (Standard Algebraic Notation) move
// Returns true if move was successfully parsed and applied
bool parse_and_apply_san(BitboardState &board, const std::string &san_move) {
  // Remove check/checkmate markers
  std::string move = san_move;
  if (!move.empty() && (move.back() == '+' || move.back() == '#')) {
    move.pop_back();
  }

  // Handle castling
  if (move == "O-O" || move == "0-0" || move == "o-o") {
    // Kingside castling
    bool is_white = board.white_to_move();
    int king_rank = is_white ? 0 : 7;
    int king_from = king_rank * 8 + 4; // e1 or e8
    int king_to = king_rank * 8 + 6;   // g1 or g8
    board.make_move(Move(king_from, king_to));
    return true;
  } else if (move == "O-O-O" || move == "0-0-0" || move == "o-o-o") {
    // Queenside castling
    bool is_white = board.white_to_move();
    int king_rank = is_white ? 0 : 7;
    int king_from = king_rank * 8 + 4; // e1 or e8
    int king_to = king_rank * 8 + 2;   // c1 or c8
    board.make_move(Move(king_from, king_to));
    return true;
  }

  // Parse piece type (K, Q, R, B, N, or pawn if empty)
  Piece piece_type = EMPTY;
  int piece_idx = 0;
  if (move.length() > 0 && std::isupper(move[0])) {
    char p = std::toupper(move[0]);
    bool is_white = board.white_to_move();
    switch (p) {
    case 'K':
      piece_type = is_white ? W_KING : B_KING;
      break;
    case 'Q':
      piece_type = is_white ? W_QUEEN : B_QUEEN;
      break;
    case 'R':
      piece_type = is_white ? W_ROOK : B_ROOK;
      break;
    case 'B':
      piece_type = is_white ? W_BISHOP : B_BISHOP;
      break;
    case 'N':
      piece_type = is_white ? W_KNIGHT : B_KNIGHT;
      break;
    default:
      return false;
    }
    piece_idx = 1;
  } else {
    // Pawn move
    bool is_white = board.white_to_move();
    piece_type = is_white ? W_PAWN : B_PAWN;
  }

  // Check for capture
  bool is_capture = false;
  if (piece_idx < static_cast<int>(move.length()) && move[piece_idx] == 'x') {
    is_capture = true;
    piece_idx++;
  }

  // Parse promotion (e.g., e8=Q)
  Piece promotion = EMPTY;
  size_t eq_pos = move.find('=');
  if (eq_pos != std::string::npos && eq_pos + 1 < move.length()) {
    char prom_char = std::toupper(move[eq_pos + 1]);
    bool is_white = board.white_to_move();
    switch (prom_char) {
    case 'Q':
      promotion = is_white ? W_QUEEN : B_QUEEN;
      break;
    case 'R':
      promotion = is_white ? W_ROOK : B_ROOK;
      break;
    case 'B':
      promotion = is_white ? W_BISHOP : B_BISHOP;
      break;
    case 'N':
      promotion = is_white ? W_KNIGHT : B_KNIGHT;
      break;
    default:
      return false;
    }
    move = move.substr(0, eq_pos);
  }

  // Parse target square (last 2 characters, e.g., "e4", "d5")
  if (move.length() < 2)
    return false;

  std::string target_str = move.substr(move.length() - 2);
  if (target_str[0] < 'a' || target_str[0] > 'h' || target_str[1] < '1' ||
      target_str[1] > '8') {
    return false;
  }

  int target_file = target_str[0] - 'a';
  int target_rank = target_str[1] - '1';
  int target_square = target_rank * 8 + target_file;

  // Find source square
  // Generate all legal moves and find matching one
  std::vector<Move> legal_moves = board.generate_legal_moves();

  // Filter moves by target square and piece type
  std::vector<Move> candidates;
  for (const Move &m : legal_moves) {
    if (m.to == target_square) {
      Piece moved_piece = board.get_piece_at(m.from);
      if (std::abs(moved_piece) == std::abs(piece_type) ||
          (piece_type == (board.white_to_move() ? W_PAWN : B_PAWN) &&
           std::abs(moved_piece) == std::abs(W_PAWN))) {
        if (promotion == EMPTY || m.promotion == promotion) {
          candidates.push_back(m);
        }
      }
    }
  }

  // Disambiguation handling (e.g., Nbd2, R1e1)
  if (candidates.size() > 1 &&
      piece_idx < static_cast<int>(move.length() - 2)) {
    std::string disambig =
        move.substr(piece_idx, move.length() - piece_idx - 2);
    std::vector<Move> filtered;
    for (const Move &m : candidates) {
      bool matches = true;
      for (char c : disambig) {
        if (std::isdigit(c)) {
          // Rank disambiguation (e.g., R1e1)
          int rank = c - '1';
          if (m.from / 8 != rank) {
            matches = false;
            break;
          }
        } else if (std::isalpha(c)) {
          // File disambiguation (e.g., Nbd2)
          int file = std::tolower(c) - 'a';
          if (m.from % 8 != file) {
            matches = false;
            break;
          }
        }
      }
      if (matches) {
        filtered.push_back(m);
      }
    }
    if (!filtered.empty()) {
      candidates = filtered;
    }
  }

  // Apply the first matching move
  if (!candidates.empty()) {
    board.make_move(candidates[0]);
    return true;
  }

  return false;
}

std::string pgn_to_fen(const std::string &pgn) {
  // Start with initial position
  BitboardState board(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  // Simple PGN parser - extract move notation
  // Remove headers and comments
  std::string moves_text = pgn;

  // Remove headers (lines starting with [)
  std::istringstream iss(moves_text);
  std::string line;
  std::string clean_moves;
  bool in_headers = true;

  while (std::getline(iss, line)) {
    if (line.empty()) {
      in_headers = false;
      continue;
    }
    if (in_headers && (line[0] == '[' || line.find("[Event") == 0 ||
                       line.find("[Site") == 0 || line.find("[Date") == 0 ||
                       line.find("[White") == 0 || line.find("[Black") == 0 ||
                       line.find("[Result") == 0)) {
      continue;
    }
    if (!in_headers || line[0] != '[') {
      clean_moves += line + " ";
      in_headers = false;
    }
  }

  // Extract move numbers and SAN moves
  // Pattern: move_number. SAN move (e.g., "1. e4 e5 2. Nf3 Nc6")
  std::regex move_pattern(R"(\d+\.\s*([^\s]+\s+[^\s]+|[^\s]+))");
  std::smatch matches;
  std::string::const_iterator search_start(clean_moves.cbegin());

  std::vector<std::string> parsed_moves;
  while (std::regex_search(search_start, clean_moves.cend(), matches,
                           move_pattern)) {
    std::string move_pair = matches[1].str();

    // Split white and black moves
    std::istringstream move_stream(move_pair);
    std::string white_move, black_move;
    move_stream >> white_move;
    if (move_stream >> black_move) {
      parsed_moves.push_back(white_move);
      parsed_moves.push_back(black_move);
    } else {
      parsed_moves.push_back(white_move);
    }

    search_start = matches[0].second;
  }

  // Also try simple space-separated moves (for PGN without move numbers)
  if (parsed_moves.empty()) {
    std::istringstream move_stream(clean_moves);
    std::string move;
    while (move_stream >> move) {
      // Skip move numbers, result markers, etc.
      if (move.find('.') == std::string::npos && move != "1-0" &&
          move != "0-1" && move != "1/2-1/2" && move != "*" &&
          move.find('-') == std::string::npos) {
        parsed_moves.push_back(move);
      }
    }
  }

  // Apply moves to board
  for (const std::string &move : parsed_moves) {
    if (!parse_and_apply_san(board, move)) {
      // If move parsing fails, try to continue or return current FEN
      // For robustness, we'll continue with next move
      continue;
    }
  }

  // Return final FEN
  return board.to_fen();
}

// ============================================================================
// MULTI-PV SEARCH IMPLEMENTATION
// ============================================================================

std::vector<PVLine>
multi_pv_search(BitboardState board, int depth, int num_lines,
                const std::function<int(const std::string &)> &evaluate,
                TranspositionTable *tt, int num_threads, KillerMoves *killers,
                HistoryTable *history, CounterMoveTable *counters) {
  depth = enforce_depth_limit(depth);

  std::vector<PVLine> pv_lines;

  // Create local instances if needed
  TranspositionTable local_tt;
  KillerMoves local_killers;
  HistoryTable local_history;
  CounterMoveTable local_counters;

  if (!tt)
    tt = &local_tt;
  if (!killers)
    killers = &local_killers;
  if (!history)
    history = &local_history;
  if (!counters)
    counters = &local_counters;

  EvalProvider eval_provider = MakeEvalProvider(evaluate, true);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    return pv_lines; // No moves available
  }

  bool maximizing = board.white_to_move();

  // Score all moves
  struct ScoredMove {
    Move move;
    BitboardState board_after;
    int score;
  };

  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(legal_moves.size());

  // Parallelize move evaluation if num_threads > 1
  if (num_threads > 1 && legal_moves.size() > 1) {
    // Release GIL before spawning threads
    nb::gil_scoped_release gil_release;

    std::vector<std::future<ScoredMove>> futures;

    auto evaluate_move = [&](const Move &move) -> ScoredMove {
      BitboardState local_board = board;
      local_board.make_move(move);
      BitboardState after_board = local_board;

      KillerMoves thread_killers = *killers;
      HistoryTable thread_history = *history;
      CounterMoveTable thread_counters = *counters;
      ContinuationHistory thread_cont_history;
      Piece moving_piece = board.get_piece_at(move.from);

      int score = alpha_beta_internal(
          local_board, depth - 1, MIN, MAX, !maximizing, eval_provider, *tt,
          true, &thread_killers, 1, &thread_history, true, &thread_counters,
          &thread_cont_history, moving_piece, move.to);

      return {move, std::move(after_board), score};
    };

    // Launch threads
    int max_parallel =
        (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    if (max_parallel == 0)
      max_parallel = 4;

    for (size_t i = 0; i < legal_moves.size(); i++) {
      // Wait for a thread to finish if we're at capacity
      while (futures.size() >= static_cast<size_t>(max_parallel)) {
        for (auto it = futures.begin(); it != futures.end();) {
          if (it->wait_for(std::chrono::microseconds(100)) ==
              std::future_status::ready) {
            scored_moves.push_back(it->get());
            it = futures.erase(it);
          } else {
            ++it;
          }
        }
      }

      futures.push_back(
          std::async(std::launch::async, evaluate_move, legal_moves[i]));
    }

    // Collect remaining results
    for (auto &future : futures) {
      scored_moves.push_back(future.get());
    }
  } else {
    // Sequential evaluation (original code)
    for (const Move &move : legal_moves) {
      BitboardState local_board = board;
      local_board.make_move(move);
      BitboardState after_board = local_board;

      KillerMoves local_killers = *killers;
      HistoryTable local_history = *history;
      CounterMoveTable local_counters = *counters;
      ContinuationHistory local_cont_history;
      Piece moving_piece = board.get_piece_at(move.from);

      int score = alpha_beta_internal(
          local_board, depth - 1, MIN, MAX, !maximizing, eval_provider, *tt,
          true, &local_killers, 1, &local_history, true, &local_counters,
          &local_cont_history, moving_piece, move.to);

      scored_moves.push_back({move, std::move(after_board), score});
    }
  }

  // Sort moves by score
  std::sort(scored_moves.begin(), scored_moves.end(),
            [maximizing](const ScoredMove &a, const ScoredMove &b) {
              if (maximizing) {
                return a.score > b.score; // Best first
              } else {
                return a.score < b.score; // Worst (best for black) first
              }
            });

  // Return top N lines
  int lines_to_return = std::min(num_lines, (int)scored_moves.size());

  for (int i = 0; i < lines_to_return; i++) {
    PVLine line;

    // Convert move to UCI
    const Move &m = scored_moves[i].move;
    char from_file = 'a' + (m.from % 8);
    char from_rank = '1' + (m.from / 8);
    char to_file = 'a' + (m.to % 8);
    char to_rank = '1' + (m.to / 8);

    line.uci_move = std::string() + from_file + from_rank + to_file + to_rank;

    // Add promotion
    if (m.promotion != EMPTY) {
      char promo = ' ';
      int piece_type = m.promotion < 0 ? -m.promotion : m.promotion;
      switch (piece_type) {
      case 2:
        promo = 'n';
        break;
      case 3:
        promo = 'b';
        break;
      case 4:
        promo = 'r';
        break;
      case 5:
        promo = 'q';
        break;
      }
      if (promo != ' ')
        line.uci_move += promo;
    }

    line.score = scored_moves[i].score;
    line.depth = depth;

    // Extract full PV from TT by following best moves
    line.pv = line.uci_move;
    BitboardState pv_board = scored_moves[i].board_after;

    // Walk through TT to extract continuation
    for (int pv_depth = 0; pv_depth < depth - 1 && pv_depth < 10; pv_depth++) {
      uint16_t pv_move_code = tt->get_best_move(pv_board.zobrist());
      if (pv_move_code == 0) {
        break;
      }

      Move pv_move = pv_board.decode_move(pv_move_code);
      pv_board.make_move(pv_move);
      line.pv += " " + pv_board.move_to_uci(pv_move_code);
    }

    pv_lines.push_back(line);
  }

  return pv_lines;
}

std::vector<PVLine>
multi_pv_search(const std::string &fen, int depth, int num_lines,
                const std::function<int(const std::string &)> &evaluate,
                TranspositionTable *tt, int num_threads, KillerMoves *killers,
                HistoryTable *history, CounterMoveTable *counters) {
  BitboardState board(fen);
  return multi_pv_search(board, depth, num_lines, evaluate, tt, num_threads,
                         killers, history, counters);
}
