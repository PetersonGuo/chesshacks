#include "functions.h"
#include "chess_board.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <thread>
#include <vector>

// ============================================================================
// KILLER MOVES IMPLEMENTATION
// ============================================================================

void KillerMoves::store(int ply, const std::string &move_fen) {
  if (ply >= MAX_DEPTH) return;
  
  // Don't store duplicates
  if (killers[ply][0] == move_fen) return;
  
  // Shift: second killer becomes first, new move becomes second
  killers[ply][1] = killers[ply][0];
  killers[ply][0] = move_fen;
}

bool KillerMoves::is_killer(int ply, const std::string &move_fen) const {
  if (ply >= MAX_DEPTH) return false;
  return killers[ply][0] == move_fen || killers[ply][1] == move_fen;
}

void KillerMoves::clear() {
  for (int i = 0; i < MAX_DEPTH; i++) {
    killers[i][0] = "";
    killers[i][1] = "";
  }
}

// ============================================================================
// HISTORY HEURISTIC IMPLEMENTATION
// ============================================================================

void HistoryTable::update(int piece, int to_square, int depth) {
  if (to_square < 0 || to_square >= 64) return;
  int idx = piece + 6; // Map -6..6 to 0..12
  if (idx < 0 || idx >= 13) return;
  
  // Reward based on depth (deeper moves are more valuable)
  history[idx][to_square] += depth * depth;
  
  // Cap to prevent overflow
  if (history[idx][to_square] > 100000) {
    age(); // Age all values when one gets too high
  }
}

int HistoryTable::get_score(int piece, int to_square) const {
  if (to_square < 0 || to_square >= 64) return 0;
  int idx = piece + 6;
  if (idx < 0 || idx >= 13) return 0;
  return history[idx][to_square];
}

void HistoryTable::clear() {
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 64; j++) {
      history[i][j] = 0;
    }
  }
}

void HistoryTable::age() {
  // Divide all values by 2 to favor recent history
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 64; j++) {
      history[i][j] /= 2;
    }
  }
}

// ============================================================================
// TRANSPOSITION TABLE IMPLEMENTATION
// ============================================================================
void TranspositionTable::store(const std::string &fen, int depth, int score,
                               TTEntryType type,
                               const std::string &best_move_fen) {
  std::lock_guard<std::mutex> lock(mutex);
  // Only store if this is a deeper search or doesn't exist
  auto it = table.find(fen);
  if (it == table.end() || it->second.depth <= depth) {
    table[fen] = {depth, score, type, best_move_fen};
  }
}

bool TranspositionTable::probe(const std::string &fen, int depth, int alpha,
                               int beta, int &score,
                               std::string &best_move_fen) {
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

  // Return best move for move ordering
  best_move_fen = entry.best_move_fen;

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

// ============================================================================
// MOVE ORDERING HELPERS
// ============================================================================

// MVV-LVA: Most Valuable Victim - Least Valuable Attacker
int get_piece_value(Piece piece) {
  int abs_piece = piece < 0 ? -piece : piece;
  static const int piece_values[] = {0,   100, 320,  330,
                                     500, 900, 20000}; // None, P, N, B, R, Q, K
  return piece_values[abs_piece];
}

int mvv_lva_score(const ChessBoard &board, const Move &move) {
  Piece victim = board.get_piece_at(move.to);
  if (victim == EMPTY)
    return 0;

  Piece attacker = board.get_piece_at(move.from);
  // Victim value * 10 - attacker value (prefer capturing with low-value pieces)
  return get_piece_value(victim) * 10 - get_piece_value(attacker);
}

// Order moves for better alpha-beta pruning
void order_moves(ChessBoard &board, std::vector<Move> &moves,
                 TranspositionTable *tt, const std::string &fen,
                 KillerMoves *killers = nullptr, int ply = 0,
                 HistoryTable *history = nullptr) {
  std::vector<std::pair<Move, int>> scored_moves;
  scored_moves.reserve(moves.size());

  // Get best move from TT if available
  std::string tt_best_move = "";
  if (tt) {
    int dummy_score;
    tt->probe(fen, 0, MIN, MAX, dummy_score, tt_best_move);
  }

  for (const Move &move : moves) {
    int score = 0;

    // Check if this is the TT best move (highest priority)
    board.make_move(move);
    std::string move_fen = board.to_fen();
    board.unmake_move(move);

    if (!tt_best_move.empty() && move_fen == tt_best_move) {
      score = 1000000; // TT move gets highest priority
    } else if (move.promotion != EMPTY) {
      // Promotions (prioritize queen promotions)
      score = 900000 + get_piece_value(move.promotion);
    } else if (killers && killers->is_killer(ply, move_fen)) {
      // Killer moves (good quiet moves from sibling nodes)
      score = 850000;
    } else if (board.is_capture(move)) {
      // Captures ordered by MVV-LVA
      score = 800000 + mvv_lva_score(board, move);
    } else {
      // Quiet moves - use history heuristic if available
      if (history) {
        Piece piece = board.get_piece_at(move.from);
        score = history->get_score(piece, move.to);
      } else {
        // Fallback to center control heuristic
        int to_rank = move.to / 8;
        int to_file = move.to % 8;
        double center_dist = std::abs(to_rank - 3.5) + std::abs(to_file - 3.5);
        score = 10000 - static_cast<int>(center_dist * 1000);
      }
    }

    scored_moves.push_back({move, score});
  }

  // Sort by score (highest first)
  std::sort(scored_moves.begin(), scored_moves.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  // Replace with sorted moves
  moves.clear();
  for (const auto &pair : scored_moves) {
    moves.push_back(pair.first);
  }
}

// ============================================================================
// QUIESCENCE SEARCH
// ============================================================================

// Quiescence search to avoid horizon effect (only search captures)
int quiescence_search(ChessBoard &board, int alpha, int beta,
                      bool maximizingPlayer,
                      const std::function<int(const std::string &)> &evaluate,
                      int q_depth = 0, int max_q_depth = 4) {
  // Limit quiescence depth to prevent infinite recursion
  if (q_depth >= max_q_depth) {
    return evaluate(board.to_fen());
  }
  
  // Stand-pat evaluation
  std::string fen = board.to_fen();
  int stand_pat = evaluate(fen);

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
      int eval = quiescence_search(board, alpha, beta, false, evaluate, q_depth + 1, max_q_depth);
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
      int eval = quiescence_search(board, alpha, beta, true, evaluate, q_depth + 1, max_q_depth);
      board.unmake_move(move);      minEval = std::min(minEval, eval);
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
int alpha_beta_basic(const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer,
                     const std::function<int(const std::string &)> &evaluate) {
  if (depth == 0) {
    return evaluate(fen);
  }

  ChessBoard board(fen);
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
      board.make_move(move);
      std::string child_fen = board.to_fen();
      int eval =
          alpha_beta_basic(child_fen, depth - 1, alpha, beta, false, evaluate);
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
      board.make_move(move);
      std::string child_fen = board.to_fen();
      int eval =
          alpha_beta_basic(child_fen, depth - 1, alpha, beta, true, evaluate);
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

// ============================================================================
// 2. OPTIMIZED VERSION - With TT, move ordering, quiescence, and optional
// multithreading
// ============================================================================

// Internal recursive function with all optimizations
// ============================================================================
// OPTIMIZED ALPHA-BETA WITH ALL FEATURES
// ============================================================================

static int
alpha_beta_internal(const std::string &fen, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable &tt, bool use_quiescence = true,
                    KillerMoves *killers = nullptr, int ply = 0,
                    HistoryTable *history = nullptr, bool allow_null_move = true) {
  // Check transposition table
  int cached_score;
  std::string tt_best_move;
  if (tt.probe(fen, depth, alpha, beta, cached_score, tt_best_move)) {
    return cached_score;
  }

  ChessBoard board(fen);

  // Null move pruning (skip if in check or zugzwang-prone position)
  if (allow_null_move && depth >= 3 && !board.is_check()) {
    // Check if we're not in an endgame (simple heuristic: need enough material)
    std::vector<Move> all_moves = board.generate_legal_moves();
    if (all_moves.size() > 3) { // Not in severe endgame
      // Make null move (skip turn)
      int R = 2; // Reduction factor
      std::string null_fen = fen; // Simplified: in real chess, would flip side to move
      
      // Try shallow search with null move
      int null_score = -alpha_beta_internal(null_fen, depth - 1 - R, -beta, -beta + 1,
                                           !maximizingPlayer, evaluate, tt, use_quiescence,
                                           killers, ply + 1, history, false);
      
      if (null_score >= beta) {
        return beta; // Null move cutoff
      }
    }
  }

  // At depth 0, enter quiescence search to avoid horizon effect
  if (depth == 0) {
    if (use_quiescence) {
      int q_score =
          quiescence_search(board, alpha, beta, maximizingPlayer, evaluate);
      tt.store(fen, 0, q_score, EXACT, "");
      return q_score;
    } else {
      int score = evaluate(fen);
      tt.store(fen, 0, score, EXACT, "");
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
    tt.store(fen, depth, score, EXACT, "");
    return score;
  }

  // Move ordering with TT, killers, MVV-LVA, history, promotions
  order_moves(board, legal_moves, &tt, fen, killers, ply, history);

  int original_alpha = alpha;
  std::string best_move_fen = "";
  int moves_searched = 0;

  if (maximizingPlayer) {
    int maxEval = MIN;
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      
      int eval;
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);
      
      // Late Move Reduction (LMR)
      // Reduce depth for later moves that are less likely to be best
      bool do_lmr = false;
      int reduction = 0;
      
      if (depth >= 3 && moves_searched >= 4 && !is_capture && !is_promotion && !board.is_check()) {
        do_lmr = true;
        // Reduce more for later moves
        reduction = 1;
        if (moves_searched >= 8) reduction = 2;
      }
      
      if (do_lmr) {
        // Search with reduced depth
        eval = alpha_beta_internal(child_fen, depth - 1 - reduction, alpha, beta, false,
                                   evaluate, tt, use_quiescence, killers, ply + 1, history);
        // If reduced search fails high, re-search at full depth
        if (eval > alpha) {
          eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                     evaluate, tt, use_quiescence, killers, ply + 1, history);
        }
      } else {
        // Normal search
        eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                   evaluate, tt, use_quiescence, killers, ply + 1, history);
      }
      
      board.unmake_move(move);
      moves_searched++;

      if (eval > maxEval) {
        maxEval = eval;
        best_move_fen = child_fen;
      }
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        // Beta cutoff - store killer move and update history
        if (killers && !board.is_capture(move)) {
          killers->store(ply, child_fen);
        }
        if (history && !board.is_capture(move)) {
          Piece piece = board.get_piece_at(move.from);
          history->update(piece, move.to, depth);
        }
        break; // Beta cutoff
      }
    }

    // Store in transposition table with best move
    if (maxEval <= original_alpha) {
      tt.store(fen, depth, maxEval, UPPER_BOUND, best_move_fen);
    } else if (maxEval >= beta) {
      tt.store(fen, depth, maxEval, LOWER_BOUND, best_move_fen);
    } else {
      tt.store(fen, depth, maxEval, EXACT, best_move_fen);
    }

    return maxEval;
  } else {
    int minEval = MAX;
    int moves_searched = 0;
    
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      
      int eval;
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);
      
      // Late Move Reduction (LMR)
      bool do_lmr = false;
      int reduction = 0;
      
      if (depth >= 3 && moves_searched >= 4 && !is_capture && !is_promotion && !board.is_check()) {
        do_lmr = true;
        reduction = 1;
        if (moves_searched >= 8) reduction = 2;
      }
      
      if (do_lmr) {
        // Search with reduced depth
        eval = alpha_beta_internal(child_fen, depth - 1 - reduction, alpha, beta, true,
                                   evaluate, tt, use_quiescence, killers, ply + 1, history);
        // If reduced search fails low, re-search at full depth
        if (eval < beta) {
          eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                     evaluate, tt, use_quiescence, killers, ply + 1, history);
        }
      } else {
        // Normal search
        eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                   evaluate, tt, use_quiescence, killers, ply + 1, history);
      }
      
      board.unmake_move(move);
      moves_searched++;

      if (eval < minEval) {
        minEval = eval;
        best_move_fen = child_fen;
      }
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        // Alpha cutoff - store killer move and update history
        if (killers && !board.is_capture(move)) {
          killers->store(ply, child_fen);
        }
        if (history && !board.is_capture(move)) {
          Piece piece = board.get_piece_at(move.from);
          history->update(piece, move.to, depth);
        }
        break; // Alpha cutoff
      }
    }

    // Store in transposition table with best move
    if (minEval <= alpha) {
      tt.store(fen, depth, minEval, UPPER_BOUND, best_move_fen);
    } else if (minEval >= beta) {
      tt.store(fen, depth, minEval, LOWER_BOUND, best_move_fen);
    } else {
      tt.store(fen, depth, minEval, EXACT, best_move_fen);
    }

    return minEval;
  }
}

// Iterative deepening wrapper with aspiration windows
int iterative_deepening(const std::string &fen, int max_depth, int alpha,
                        int beta, bool maximizingPlayer,
                        const std::function<int(const std::string &)> &evaluate,
                        TranspositionTable &tt, KillerMoves *killers = nullptr,
                        HistoryTable *history = nullptr) {
  int best_score = 0;
  const int ASPIRATION_WINDOW = 50; // Initial window size

  // Search with increasing depth
  for (int depth = 1; depth <= max_depth; depth++) {
    int current_alpha = alpha;
    int current_beta = beta;
    
    // Use aspiration windows for depth >= 3
    if (depth >= 3) {
      current_alpha = best_score - ASPIRATION_WINDOW;
      current_beta = best_score + ASPIRATION_WINDOW;
      
      // Ensure we don't go beyond original bounds
      if (current_alpha < alpha) current_alpha = alpha;
      if (current_beta > beta) current_beta = beta;
    }
    
    // Search with aspiration window
    int score = alpha_beta_internal(fen, depth, current_alpha, current_beta, maximizingPlayer,
                                    evaluate, tt, true, killers, 0, history);
    
    // If score falls outside window, re-search with full window
    if (depth >= 3 && (score <= current_alpha || score >= current_beta)) {
      // Failed low or high - re-search with full window
      score = alpha_beta_internal(fen, depth, alpha, beta, maximizingPlayer,
                                  evaluate, tt, true, killers, 0, history);
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
int alpha_beta_optimized(const std::string &fen, int depth, int alpha, int beta,
                         bool maximizingPlayer,
                         const std::function<int(const std::string &)> &evaluate,
                         TranspositionTable *tt, int num_threads,
                         KillerMoves *killers, HistoryTable *history) {
  TranspositionTable local_tt;
  TranspositionTable &tt_ref = tt ? *tt : local_tt;

  // Create local killer/history tables if not provided
  KillerMoves local_killers;
  HistoryTable local_history;
  KillerMoves *killers_ptr = killers ? killers : &local_killers;
  HistoryTable *history_ptr = history ? history : &local_history;

  // If no threading requested, use iterative deepening with single thread
  if (num_threads <= 1 || depth <= 2) {
    return iterative_deepening(fen, depth, alpha, beta, maximizingPlayer,
                               evaluate, tt_ref, killers_ptr, history_ptr);
  }

  // Parallel search at root level with iterative deepening
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4;
  }

  // First do iterative deepening up to depth-1 to populate TT
  if (depth > 1) {
    iterative_deepening(fen, depth - 1, alpha, beta, maximizingPlayer, evaluate,
                        tt_ref, killers_ptr, history_ptr);
  }

  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    if (board.is_check()) {
      return maximizingPlayer ? MIN + 1 : MAX - 1;
    } else {
      return 0;
    }
  }

  // Order root moves using TT, killer, history information
  order_moves(board, legal_moves, &tt_ref, fen, killers_ptr, 0, history_ptr);

  struct MoveScore {
    Move move;
    int score;
  };

  std::vector<std::future<MoveScore>> futures;

  auto evaluate_move = [&](const Move &move) -> MoveScore {
    ChessBoard local_board(fen);
    local_board.make_move(move);
    std::string child_fen = local_board.to_fen();

    int score;
    if (maximizingPlayer) {
      score = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                  evaluate, tt_ref, true, killers_ptr, 1, history_ptr);
    } else {
      score = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                  evaluate, tt_ref, true, killers_ptr, 1, history_ptr);
    }

    return {move, score};
  };

  // Launch threads for moves
  for (size_t i = 0; i < legal_moves.size(); i++) {
    while (futures.size() >= static_cast<size_t>(num_threads)) {
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

// ============================================================================
// 3. CUDA VERSION - Placeholder that falls back to optimized
// ============================================================================
int alpha_beta_cuda(const std::string &fen, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable *tt, KillerMoves *killers, 
                    HistoryTable *history) {
  // TODO: Implement CUDA-accelerated batch evaluation
  // For now, fall back to optimized CPU version
  return alpha_beta_optimized(fen, depth, alpha, beta, maximizingPlayer,
                              evaluate, tt, 0, killers, history);
}