#include "core/search.h"
#include "core/search_internal.h"

#include "core/evaluation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>

using bitboard::BitboardState;
using search_internal::alpha_beta_impl;
using search_internal::alpha_beta_internal;
using search_internal::enforce_depth_limit;
using search_internal::EvalProvider;
using search_internal::MakeEvalProvider;

namespace nb = nanobind;

namespace {

BitboardState apply_move_to_board(BitboardState board, uint16_t move_code) {
  if (move_code == 0) {
    return board;
  }
  BoardMove move = board.decode_move(move_code);
  board.make_move(move);
  return board;
}

std::string convert_best_move_to_uci(BitboardState board,
                                     uint16_t best_move_code) {
  if (best_move_code == 0) {
    return "";
  }
  return board.move_to_uci(best_move_code);
}

} // namespace

static BitboardState
find_best_move_impl(BitboardState board, int depth,
                    const EvalProvider &evaluate, TranspositionTable *tt,
                    int num_threads, KillerMoves *killers,
                    HistoryTable *history, CounterMoveTable *counters,
                    uint16_t *best_move_code_out = nullptr) {
  depth = enforce_depth_limit(depth);
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
  std::vector<BoardMove> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    return board;
  }

  bool maximizing = board.white_to_move();

  BitboardState search_board = board;
  alpha_beta_impl(search_board, depth, MIN, MAX, maximizing, evaluate, tt,
                  num_threads, killers, history, counters);

  uint16_t best_move_code = tt->get_best_move(root_key);

  if (best_move_code != 0) {
    if (best_move_code_out)
      *best_move_code_out = best_move_code;
    return apply_move_to_board(board, best_move_code);
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
    auto evaluate_move = [&](const BoardMove &move) -> MoveResult {
      BitboardState local_board = board;
      BoardPiece moving_piece = local_board.get_piece_at(move.from);
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
    for (const BoardMove &move : legal_moves) {
      BitboardState local_board = board;
      BoardPiece moving_piece = local_board.get_piece_at(move.from);
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
    return board;
  }

  return apply_move_to_board(board, fallback_best_code);
}

BitboardState
find_best_move(bitboard::BitboardState board, int depth,
               const std::function<int(const BitboardState &)> &evaluate,
               TranspositionTable *tt, int num_threads, KillerMoves *killers,
               HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate);
  return find_best_move_impl(board, depth, eval_provider, tt, num_threads,
                             killers, history, counters);
}

BitboardState find_best_move_builtin(bitboard::BitboardState board, int depth,
                                     TranspositionTable *tt, int num_threads,
                                     KillerMoves *killers,
                                     HistoryTable *history,
                                     CounterMoveTable *counters) {
  std::function<int(const BitboardState &)> native_eval =
      static_cast<int (*)(const BitboardState &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(native_eval);
  return find_best_move_impl(board, depth, eval_provider, tt, num_threads,
                             killers, history, counters);
}

std::string
get_best_move_uci(BitboardState board, int depth,
                  const std::function<int(const BitboardState &)> &evaluate,
                  TranspositionTable *tt, int num_threads, KillerMoves *killers,
                  HistoryTable *history, CounterMoveTable *counters) {
  EvalProvider eval_provider = MakeEvalProvider(evaluate);
  uint16_t best_move_code = 0;
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
  std::function<int(const BitboardState &)> material_eval_fn =
      static_cast<int (*)(const BitboardState &)>(&evaluate_material);
  EvalProvider eval_provider = MakeEvalProvider(material_eval_fn);
  find_best_move_impl(board, depth, eval_provider, tt, num_threads, killers,
                      history, counters, &best_move_code);
  return convert_best_move_to_uci(board, best_move_code);
}

std::vector<PVLine>
multi_pv_search(BitboardState board, int depth, int num_lines,
                const std::function<int(const BitboardState &)> &evaluate,
                TranspositionTable *tt, int num_threads, KillerMoves *killers,
                HistoryTable *history, CounterMoveTable *counters) {
  depth = enforce_depth_limit(depth);

  std::vector<PVLine> pv_lines;

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

  EvalProvider eval_provider = MakeEvalProvider(evaluate);
  std::vector<BoardMove> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    return pv_lines;
  }

  bool maximizing = board.white_to_move();

  struct ScoredMove {
    BoardMove move;
    BitboardState board_after;
    int score;
  };

  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(legal_moves.size());

  if (num_threads > 1 && legal_moves.size() > 1) {
    nb::gil_scoped_release gil_release;
    std::vector<ScoredMove> parallel_results(legal_moves.size());
    std::vector<char> evaluated(legal_moves.size(), 0);
    std::atomic<size_t> next_index{0};
    int threads_to_use = std::max(
        1, std::min(num_threads == 0
                        ? static_cast<int>(std::thread::hardware_concurrency())
                        : num_threads,
                    static_cast<int>(legal_moves.size())));

    auto worker = [&](int /*thread_id*/) {
      while (true) {
        size_t idx = next_index.fetch_add(1);
        if (idx >= legal_moves.size())
          break;

        BitboardState local_board = board;
        const BoardMove &move = legal_moves[idx];
        local_board.make_move(move);
        BitboardState after_board = local_board;

        KillerMoves thread_killers = *killers;
        HistoryTable thread_history = *history;
        CounterMoveTable thread_counters = *counters;
        ContinuationHistory thread_cont_history;
        BoardPiece moving_piece = board.get_piece_at(move.from);

        int score = alpha_beta_internal(
            local_board, depth - 1, MIN, MAX, !maximizing, eval_provider, *tt,
            true, &thread_killers, 1, &thread_history, true, &thread_counters,
            &thread_cont_history, moving_piece, move.to);

        parallel_results[idx] = {move, std::move(after_board), score};
        evaluated[idx] = 1;
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(std::max(0, threads_to_use - 1));
    for (int t = 1; t < threads_to_use; ++t) {
      workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &th : workers) {
      th.join();
    }

    for (size_t i = 0; i < parallel_results.size(); ++i) {
      if (!evaluated[i])
        continue;
      scored_moves.push_back(parallel_results[i]);
    }
  } else {
    for (const BoardMove &move : legal_moves) {
      BitboardState local_board = board;
      local_board.make_move(move);
      BitboardState after_board = local_board;

      KillerMoves local_killers = *killers;
      HistoryTable local_history = *history;
      CounterMoveTable local_counters = *counters;
      ContinuationHistory local_cont_history;
      BoardPiece moving_piece = board.get_piece_at(move.from);

      int score = alpha_beta_internal(
          local_board, depth - 1, MIN, MAX, !maximizing, eval_provider, *tt,
          true, &local_killers, 1, &local_history, true, &local_counters,
          &local_cont_history, moving_piece, move.to);

      scored_moves.push_back({move, std::move(after_board), score});
    }
  }

  std::sort(scored_moves.begin(), scored_moves.end(),
            [maximizing](const ScoredMove &a, const ScoredMove &b) {
              if (maximizing) {
                return a.score > b.score;
              }
              return a.score < b.score;
            });

  int lines_to_return = std::min(num_lines, (int)scored_moves.size());

  for (int i = 0; i < lines_to_return; i++) {
    PVLine line;

    const BoardMove &m = scored_moves[i].move;
    char from_file = 'a' + (m.from % 8);
    char from_rank = '1' + (m.from / 8);
    char to_file = 'a' + (m.to % 8);
    char to_rank = '1' + (m.to / 8);

    line.uci_move = std::string() + from_file + from_rank + to_file + to_rank;

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

    line.pv = line.uci_move;
    BitboardState pv_board = scored_moves[i].board_after;

    for (int pv_depth = 0; pv_depth < depth - 1 && pv_depth < 10; pv_depth++) {
      uint16_t pv_move_code = tt->get_best_move(pv_board.zobrist());
      if (pv_move_code == 0) {
        break;
      }

      BoardMove pv_move = pv_board.decode_move(pv_move_code);
      pv_board.make_move(pv_move);
      line.pv += " " + pv_board.move_to_uci(pv_move_code);
    }

    pv_lines.push_back(line);
  }

  return pv_lines;
}
