#include "core/search.h"
#include "core/search_internal.h"

#include "core/evaluation.h"
#include "core/move_ordering.h"
#include "core/transposition_table.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>

#ifdef BUILDING_PY_MODULE
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#else
namespace nb {
class gil_scoped_release {
public:
  gil_scoped_release() = default;
};
} // namespace nb
#endif
namespace {

class ThreadPool {
public:
  static ThreadPool &Instance() {
    static ThreadPool pool;
    return pool;
  }

  void run(size_t workers, const std::function<void()> &task,
           const std::function<void()> &main_task) {
    if (workers == 0) {
      main_task();
      return;
    }
    ensure_threads(workers);
    {
      std::lock_guard<std::mutex> lk(mutex_);
      task_fn_ = task;
      pending_workers_ = std::min(workers, threads_.size());
      outstanding_.store(pending_workers_);
      task_generation_++;
      task_active_ = true;
    }
    cv_.notify_all();
    main_task();
    std::unique_lock<std::mutex> lk(mutex_);
    done_cv_.wait(lk, [&] { return !task_active_; });
  }

private:
  ThreadPool() = default;

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    for (auto &thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void ensure_threads(size_t count) {
    while (threads_.size() < count) {
      size_t id = threads_.size();
      threads_.emplace_back([this, id] { worker_loop(id); });
    }
  }

  void worker_loop(size_t /*id*/) {
    size_t seen_generation = 0;
    while (true) {
      std::unique_lock<std::mutex> lk(mutex_);
      cv_.wait(lk, [&] {
        return shutdown_ || task_generation_ > seen_generation;
      });
      if (shutdown_)
        break;
      if (!task_active_ || pending_workers_ == 0) {
        seen_generation = task_generation_;
        continue;
      }
      pending_workers_--;
      auto task = task_fn_;
      seen_generation = task_generation_;
      lk.unlock();

      task();

      if (outstanding_.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> lk2(mutex_);
        task_active_ = false;
        done_cv_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  bool shutdown_ = false;
  bool task_active_ = false;
  size_t task_generation_ = 0;
  size_t pending_workers_ = 0;
  std::function<void()> task_fn_;
  std::atomic<size_t> outstanding_{0};
};

} // namespace

#ifdef CUDA_ENABLED
#include "cuda/cuda_utils.h"
#endif

using bitboard::BitboardState;
using search_internal::EvalProvider;
using search_internal::MakeEvalProvider;

// ============================================================================
// Alpha-beta search with TT, move ordering, quiescence, and optional threading
// ============================================================================

namespace search_internal {

int alpha_beta_internal(BitboardState &board, int depth, int alpha, int beta,
                        bool maximizingPlayer, const EvalProvider &evaluate,
                        TranspositionTable &tt, bool use_quiescence,
                        KillerMoves *killers, int ply, HistoryTable *history,
                        bool allow_null_move, CounterMoveTable *counters,
                        ContinuationHistory *cont_history, int prev_piece,
                        int prev_to) {
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

  std::vector<BoardMove> legal_moves = board.generate_legal_moves();

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
  if (tt_best_move == 0 && depth >= 5) {
    int iid_depth = depth - 2;
    BitboardState iid_board = board;
    alpha_beta_internal(iid_board, iid_depth, alpha, beta, maximizingPlayer,
                        evaluate, tt, use_quiescence, killers, ply, history,
                        allow_null_move, counters, cont_history, prev_piece,
                        prev_to);
    tt.probe(board_key, iid_depth, alpha, beta, cached_score, tt_best_move);
  }

  // Singular Extensions
  int extension = 0;
  if (tt_best_move != 0 && depth >= 6) {
    int singular_beta = cached_score - depth * 2;
    int singular_depth = depth / 2;

    int best_alternative = MIN;
    for (const BoardMove &move : legal_moves) {
      if (move.encoded == tt_best_move)
        continue;
      BoardPiece singular_piece = board.get_piece_at(move.from);
      BitboardState alt_board = board;
      alt_board.make_move(move);
      int score = alpha_beta_internal(
          alt_board, singular_depth, singular_beta - 1, singular_beta,
          !maximizingPlayer, evaluate, tt, use_quiescence, killers, ply + 1,
          history, true, counters, cont_history, singular_piece, move.to);

      best_alternative = std::max(best_alternative, score);

      if (best_alternative >= singular_beta) {
        break;
      }
    }

    if (best_alternative < singular_beta) {
      extension = 1;
    }
  }

  order_moves(board, legal_moves, tt_best_move, killers, ply, history, counters,
              prev_piece, prev_to, cont_history);

  int original_alpha = alpha;
  uint16_t best_move_code = 0;
  int moves_searched = 0;

  if (maximizingPlayer) {
    int maxEval = MIN;

    bool use_futility = false;
    int futility_margin = 0;
    int static_eval = 0;

    if (depth <= 6 && !board.is_check()) {
      static_eval = evaluate(board);
      futility_margin = 100 + 50 * depth;
      use_futility = true;
    }

    for (const BoardMove &move : legal_moves) {
      BoardPiece moving_piece = board.get_piece_at(move.from);
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);
      bool is_quiet = !is_capture && !is_promotion;
      board.make_move(move);
      int eval;

      if (use_futility && is_quiet && moves_searched > 0 && !board.is_check()) {
        if (static_eval + futility_margin < alpha) {
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

    for (const BoardMove &move : legal_moves) {
      BoardPiece moving_piece = board.get_piece_at(move.from);
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
}

int iterative_deepening(BitboardState &root_board, int max_depth, int alpha,
                        int beta, bool maximizingPlayer,
                        const EvalProvider &evaluate, TranspositionTable &tt,
                        KillerMoves *killers, HistoryTable *history,
                        CounterMoveTable *counters,
                        ContinuationHistory *cont_history) {
  max_depth = enforce_depth_limit(max_depth);
  int best_score = 0;
  const int ASPIRATION_WINDOW = 50;

  for (int depth = 1; depth <= max_depth; depth++) {
    BitboardState search_board = root_board;
    int current_alpha = alpha;
    int current_beta = beta;

    if (depth >= 3) {
      current_alpha = best_score - ASPIRATION_WINDOW;
      current_beta = best_score + ASPIRATION_WINDOW;

      if (current_alpha < alpha)
        current_alpha = alpha;
      if (current_beta > beta)
        current_beta = beta;
    }

    int score =
        alpha_beta_internal(search_board, depth, current_alpha, current_beta,
                            maximizingPlayer, evaluate, tt, true, killers, 0,
                            history, true, counters, cont_history, 0, -1);

    if (depth >= 3 && (score <= current_alpha || score >= current_beta)) {
      BitboardState retry_board = root_board;
      score = alpha_beta_internal(
          retry_board, depth, alpha, beta, maximizingPlayer, evaluate, tt, true,
          killers, 0, history, true, counters, cont_history, 0, -1);
    }

    best_score = score;

    if (best_score <= MIN + 1000 || best_score >= MAX - 1000) {
      break;
    }
  }

  return best_score;
}

int alpha_beta_impl(BitboardState board, int depth, int alpha, int beta,
                    bool maximizingPlayer, const EvalProvider &evaluate,
                    TranspositionTable *tt, int num_threads,
                    KillerMoves *killers, HistoryTable *history,
                    CounterMoveTable *counters) {
  depth = search_internal::enforce_depth_limit(depth);
  TranspositionTable local_tt;
  TranspositionTable &tt_ref = tt ? *tt : local_tt;

  KillerMoves local_killers;
  HistoryTable local_history;
  CounterMoveTable local_counters;
  ContinuationHistory local_cont_history;
  KillerMoves *killers_ptr = killers ? killers : &local_killers;
  HistoryTable *history_ptr = history ? history : &local_history;
  CounterMoveTable *counters_ptr = counters ? counters : &local_counters;
  ContinuationHistory *cont_history_ptr = &local_cont_history;

  const uint64_t root_key = board.zobrist();
  std::vector<BoardMove> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    if (board.is_check()) {
      return maximizingPlayer ? MIN + 1 : MAX - 1;
    }
    return 0;
  }

  unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
  bool auto_threads = (num_threads <= 0);

  auto capped_auto_threads = [&](int depth) -> int {
    unsigned int depth_cap =
        depth <= 5 ? 4u : depth <= 7 ? 8u : std::min(16u, hw_threads);
    return static_cast<int>(std::max(1u, std::min(hw_threads, depth_cap)));
  };

  int effective_threads = auto_threads
                              ? capped_auto_threads(depth)
                              : std::max(1, std::min(num_threads, 64));
  effective_threads =
      std::min(effective_threads, static_cast<int>(legal_moves.size()));
  const int original_alpha = alpha;
  const int original_beta = beta;

  bool use_parallel = (effective_threads > 1 && depth > 2);

  if (!use_parallel) {
    return iterative_deepening(board, depth, alpha, beta, maximizingPlayer,
                               evaluate, tt_ref, killers_ptr, history_ptr,
                               counters_ptr, cont_history_ptr);
  }

  if (use_parallel && depth > 3) {
    int warmup_depth = std::max(2, depth - 2);
    BitboardState prep_board = board;
    iterative_deepening(prep_board, warmup_depth, alpha, beta, maximizingPlayer,
                        evaluate, tt_ref, killers_ptr, history_ptr,
                        counters_ptr, cont_history_ptr);
  }

  uint16_t root_tt_move = tt_ref.get_best_move(root_key);
  order_moves(board, legal_moves, root_tt_move, killers_ptr, 0, history_ptr,
              counters_ptr, 0, -1, cont_history_ptr);

  struct MoveScore {
    BoardMove move;
    int score;
  };

  std::vector<MoveScore> results(legal_moves.size());
  std::vector<char> evaluated(legal_moves.size(), 0);

  nb::gil_scoped_release gil_release;

  int threads_to_use = std::max(
      1, std::min(effective_threads, static_cast<int>(legal_moves.size())));

  auto search_move = [&](size_t idx, int local_alpha, int local_beta) -> int {
    BitboardState local_board = board;
    const BoardMove &move = legal_moves[idx];
    BoardPiece moving_piece = local_board.get_piece_at(move.from);
    local_board.make_move(move);

    KillerMoves thread_killers = *killers_ptr;
    HistoryTable thread_history = *history_ptr;
    CounterMoveTable thread_counters = *counters_ptr;
    ContinuationHistory thread_cont_history;

    return alpha_beta_internal(
        local_board, depth - 1, local_alpha, local_beta, !maximizingPlayer,
        evaluate, tt_ref, true, &thread_killers, 1, &thread_history, true,
        &thread_counters, &thread_cont_history, moving_piece, move.to);
  };

  int first_score = search_move(0, alpha, beta);
  results[0] = {legal_moves[0], first_score};
  evaluated[0] = 1;

  std::atomic<int> shared_bound(first_score);
  std::atomic<bool> cutoff(false);

  auto update_cutoff = [&](int score) {
    if (maximizingPlayer) {
      if (score >= beta) {
        cutoff.store(true, std::memory_order_relaxed);
      }
    } else {
      if (score <= alpha) {
        cutoff.store(true, std::memory_order_relaxed);
      }
    }
  };
  update_cutoff(first_score);

  if (!cutoff.load(std::memory_order_relaxed) && legal_moves.size() > 1 &&
      threads_to_use > 1) {
    std::atomic<size_t> next_index(1);

    auto worker_body = [&]() {
      while (true) {
        if (cutoff.load(std::memory_order_relaxed))
          break;
        size_t idx = next_index.fetch_add(1);
        if (idx >= legal_moves.size())
          break;

        int current_bound = shared_bound.load(std::memory_order_relaxed);
        int local_alpha =
            maximizingPlayer ? std::max(alpha, current_bound) : alpha;
        int local_beta =
            maximizingPlayer ? beta : std::min(beta, current_bound);

        int score = search_move(idx, local_alpha, local_beta);
        const bool tightened_lower = maximizingPlayer && (local_alpha > alpha);
        const bool tightened_upper = !maximizingPlayer && (local_beta < beta);

        const bool fail_low = (score <= local_alpha);
        const bool fail_high = (score >= local_beta);

        if ((tightened_lower && fail_low) || (tightened_upper && fail_high)) {
          score = search_move(idx, alpha, beta);
        }

        results[idx] = {legal_moves[idx], score};
        evaluated[idx] = 1;

        if (maximizingPlayer) {
          int prev = shared_bound.load(std::memory_order_relaxed);
          while (score > prev && !shared_bound.compare_exchange_weak(
                                     prev, score, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
          }
          if (score >= beta) {
            cutoff.store(true, std::memory_order_relaxed);
          }
        } else {
          int prev = shared_bound.load(std::memory_order_relaxed);
          while (score < prev && !shared_bound.compare_exchange_weak(
                                     prev, score, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
          }
          if (score <= alpha) {
            cutoff.store(true, std::memory_order_relaxed);
          }
        }
      }
    };

    ThreadPool::Instance().run(threads_to_use - 1, worker_body, worker_body);
  }

  int best_index = -1;
  int best_score = maximizingPlayer ? MIN : MAX;
  for (size_t i = 0; i < results.size(); ++i) {
    if (!evaluated[i])
      continue;
    if (best_index == -1) {
      best_index = static_cast<int>(i);
      best_score = results[i].score;
      continue;
    }
    if (maximizingPlayer) {
      if (results[i].score > best_score) {
        best_index = static_cast<int>(i);
        best_score = results[i].score;
      }
    } else {
      if (results[i].score < best_score) {
        best_index = static_cast<int>(i);
        best_score = results[i].score;
      }
    }
  }

  if (best_index == -1) {
    return maximizingPlayer ? MIN : MAX;
  }

  uint16_t best_move_code = results[best_index].move.encoded;

  TTEntryType entry_type = EXACT;
  if (best_score <= original_alpha) {
    entry_type = UPPER_BOUND;
  } else if (best_score >= original_beta) {
    entry_type = LOWER_BOUND;
  }

  tt_ref.store(root_key, depth, best_score, entry_type, best_move_code);
  return best_score;
}

} // namespace search_internal

int alpha_beta(BitboardState board, int depth, int alpha, int beta,
               bool maximizingPlayer,
               const std::function<int(const BitboardState &)> &evaluate,
               TranspositionTable *tt, int num_threads, KillerMoves *killers,
               HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate);
  return alpha_beta_impl(board, depth, alpha, beta, maximizingPlayer,
                         eval_provider, tt, num_threads, killers, history,
                         counters);
}

int alpha_beta_builtin(BitboardState board, int depth, int alpha, int beta,
                       bool maximizingPlayer, TranspositionTable *tt,
                       int num_threads, KillerMoves *killers,
                       HistoryTable *history, CounterMoveTable *counters) {
  std::function<int(const BitboardState &)> native_eval =
      static_cast<int (*)(const BitboardState &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(native_eval);
  return alpha_beta_impl(board, depth, alpha, beta, maximizingPlayer,
                         eval_provider, tt, num_threads, killers, history,
                         counters);
}

#ifdef CUDA_ENABLED
int alpha_beta_cuda(BitboardState board, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const BitboardState &)> &evaluate,
                    TranspositionTable *tt, KillerMoves *killers,
                    HistoryTable *history, CounterMoveTable *counters) {
  depth = search_internal::enforce_depth_limit(depth);
  if (is_cuda_available()) {
    return alpha_beta(board, depth, alpha, beta, maximizingPlayer, evaluate, tt,
                      0, killers, history, counters);
  }
  return alpha_beta(board, depth, alpha, beta, maximizingPlayer, evaluate, tt,
                    0, killers, history, counters);
}
#endif
