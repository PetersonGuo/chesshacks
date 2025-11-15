#include "functions.h"
#include "chess_board.h"
#include <algorithm>
#include <future>
#include <thread>
#include <vector>

// Transposition Table implementation (thread-safe)
void TranspositionTable::store(const std::string &fen, int depth, int score,
                               TTEntryType type) {
  std::lock_guard<std::mutex> lock(mutex);
  // Only store if this is a deeper search or doesn't exist
  auto it = table.find(fen);
  if (it == table.end() || it->second.depth <= depth) {
    table[fen] = {depth, score, type};
  }
}

bool TranspositionTable::probe(const std::string &fen, int depth, int alpha,
                               int beta, int &score) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = table.find(fen);
  if (it == table.end()) {
    return false;
  }

  const TTEntry &entry = it->second;

  // Only use if the stored depth is >= current depth
  if (entry.depth < depth) {
    return false;
  }

  // Check if we can use this cached value
  if (entry.type == EXACT) {
    score = entry.score;
    return true;
  } else if (entry.type == LOWER_BOUND && entry.score >= beta) {
    score = entry.score;
    return true;
  } else if (entry.type == UPPER_BOUND && entry.score <= alpha) {
    score = entry.score;
    return true;
  }

  return false;
}

void TranspositionTable::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  table.clear();
}

size_t TranspositionTable::size() const {
  std::lock_guard<std::mutex> lock(mutex);
  return table.size();
}

// Original alpha_beta without transposition table (backup path)
int alpha_beta(const std::string &fen, int depth, int alpha, int beta,
               bool maximizingPlayer,
               const std::function<int(const std::string &)> &evaluate) {
  if (depth == 0) {
    // Call the Python NNUE evaluation function
    return evaluate(fen);
  }

  // Create board and generate child positions
  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  // Terminal position (checkmate or stalemate)
  if (legal_moves.empty()) {
    if (board.is_check()) {
      // Checkmate - return extreme value
      return maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      // Stalemate - return draw score
      return 0;
    }
  }

  if (maximizingPlayer) {
    int maxEval = MIN;
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      int eval = alpha_beta(child_fen, depth - 1, alpha, beta, false, evaluate);
      board.unmake_move(move);

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break; // Beta cut-off
      }
    }
    return maxEval;
  } else {
    int minEval = MAX;
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      int eval = alpha_beta(child_fen, depth - 1, alpha, beta, true, evaluate);
      board.unmake_move(move);

      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        break; // Alpha cut-off
      }
    }
    return minEval;
  }
}

// Alpha-beta with transposition table (optimized path)
int alpha_beta_with_tt(const std::string &fen, int depth, int alpha, int beta,
                       bool maximizingPlayer,
                       const std::function<int(const std::string &)> &evaluate,
                       TranspositionTable &tt) {
  // Check transposition table
  int cached_score;
  if (tt.probe(fen, depth, alpha, beta, cached_score)) {
    return cached_score;
  }

  if (depth == 0) {
    // Call the Python NNUE evaluation function
    int score = evaluate(fen);
    tt.store(fen, depth, score, EXACT);
    return score;
  }

  // Create board and generate child positions
  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  // Terminal position (checkmate or stalemate)
  if (legal_moves.empty()) {
    int score;
    if (board.is_check()) {
      // Checkmate - return extreme value
      score = maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      // Stalemate - return draw score
      score = 0;
    }
    tt.store(fen, depth, score, EXACT);
    return score;
  }

  int original_alpha = alpha;

  if (maximizingPlayer) {
    int maxEval = MIN;
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      int eval = alpha_beta_with_tt(child_fen, depth - 1, alpha, beta, false,
                                    evaluate, tt);
      board.unmake_move(move);

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break; // Beta cut-off
      }
    }

    // Store in transposition table
    if (maxEval <= original_alpha) {
      tt.store(fen, depth, maxEval, UPPER_BOUND);
    } else if (maxEval >= beta) {
      tt.store(fen, depth, maxEval, LOWER_BOUND);
    } else {
      tt.store(fen, depth, maxEval, EXACT);
    }

    return maxEval;
  } else {
    int minEval = MAX;
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      int eval = alpha_beta_with_tt(child_fen, depth - 1, alpha, beta, true,
                                    evaluate, tt);
      board.unmake_move(move);

      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        break; // Alpha cut-off
      }
    }

    // Store in transposition table
    if (minEval <= alpha) {
      tt.store(fen, depth, minEval, UPPER_BOUND);
    } else if (minEval >= beta) {
      tt.store(fen, depth, minEval, LOWER_BOUND);
    } else {
      tt.store(fen, depth, minEval, EXACT);
    }

    return minEval;
  }
}

// Parallel alpha-beta without transposition table
int alpha_beta_parallel(const std::string &fen, int depth, int alpha, int beta,
                        bool maximizingPlayer,
                        const std::function<int(const std::string &)> &evaluate,
                        int num_threads) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4; // fallback
  }

  TranspositionTable tt;
  return alpha_beta_parallel_with_tt(fen, depth, alpha, beta, maximizingPlayer,
                                     evaluate, tt, num_threads);
}

// Parallel alpha-beta with transposition table
int alpha_beta_parallel_with_tt(
    const std::string &fen, int depth, int alpha, int beta,
    bool maximizingPlayer,
    const std::function<int(const std::string &)> &evaluate,
    TranspositionTable &tt, int num_threads) {
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4; // fallback
  }

  // For shallow depths or single-threaded, use regular search
  if (depth <= 2 || num_threads == 1) {
    return alpha_beta_with_tt(fen, depth, alpha, beta, maximizingPlayer,
                              evaluate, tt);
  }

  // Generate root moves
  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  // Terminal position
  if (legal_moves.empty()) {
    if (board.is_check()) {
      return maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      return 0;
    }
  }

  // Evaluate moves in parallel
  struct MoveScore {
    Move move;
    int score;
  };

  std::vector<std::future<MoveScore>> futures;

  // Lambda to evaluate a single move
  auto evaluate_move = [&](const Move &move) -> MoveScore {
    ChessBoard local_board(fen);
    local_board.make_move(move);
    std::string child_fen = local_board.to_fen();

    int local_alpha = alpha;
    int local_beta = beta;

    // Search the position
    int score;
    if (maximizingPlayer) {
      score = alpha_beta_with_tt(child_fen, depth - 1, local_alpha, local_beta,
                                 false, evaluate, tt);
    } else {
      score = alpha_beta_with_tt(child_fen, depth - 1, local_alpha, local_beta,
                                 true, evaluate, tt);
    }

    return {move, score};
  };

  // Launch threads for moves
  for (size_t i = 0; i < legal_moves.size(); i++) {
    // Limit concurrent threads
    while (futures.size() >= static_cast<size_t>(num_threads)) {
      // Wait for at least one to finish
      for (auto it = futures.begin(); it != futures.end();) {
        if (it->wait_for(std::chrono::milliseconds(0)) ==
            std::future_status::ready) {
          it = futures.erase(it);
        } else {
          ++it;
        }
      }
      if (futures.size() >= static_cast<size_t>(num_threads)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    futures.push_back(
        std::async(std::launch::async, evaluate_move, legal_moves[i]));
  }

  // Collect results
  int best_score = maximizingPlayer ? MIN : MAX;
  for (auto &future : futures) {
    MoveScore result = future.get();
    if (maximizingPlayer) {
      best_score = std::max(best_score, result.score);
    } else {
      best_score = std::min(best_score, result.score);
    }
  }

  return best_score;
}