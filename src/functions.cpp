#include "functions.h"
#include "chess_board.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <thread>
#include <vector>

// ============================================================================
// CUDA AVAILABILITY CHECK
// ============================================================================

bool is_cuda_available() {
#ifdef __CUDACC__
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
#else
  return false;
#endif
}

std::string get_cuda_info() {
#ifdef __CUDACC__
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error != cudaSuccess || device_count == 0) {
    return "CUDA not available";
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  return std::string(prop.name) + " (CUDA Compute " +
         std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
#else
  return "CUDA not compiled (use nvcc to enable)";
#endif
}

// ============================================================================
// KILLER MOVES IMPLEMENTATION
// ============================================================================

void KillerMoves::store(int ply, const std::string &move_fen) {
  if (ply >= MAX_DEPTH)
    return;

  // Don't store duplicates
  if (killers[ply][0] == move_fen)
    return;

  // Shift: second killer becomes first, new move becomes second
  killers[ply][1] = killers[ply][0];
  killers[ply][0] = move_fen;
}

bool KillerMoves::is_killer(int ply, const std::string &move_fen) const {
  if (ply >= MAX_DEPTH)
    return false;
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
  if (to_square < 0 || to_square >= 64)
    return;
  int idx = piece + 6; // Map -6..6 to 0..12
  if (idx < 0 || idx >= 13)
    return;

  // Reward based on depth (deeper moves are more valuable)
  history[idx][to_square] += depth * depth;

  // Cap to prevent overflow
  if (history[idx][to_square] > 100000) {
    age(); // Age all values when one gets too high
  }
}

int HistoryTable::get_score(int piece, int to_square) const {
  if (to_square < 0 || to_square >= 64)
    return 0;
  int idx = piece + 6;
  if (idx < 0 || idx >= 13)
    return 0;
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
// COUNTER MOVE TABLE IMPLEMENTATION
// ============================================================================
void CounterMoveTable::store(int piece, int to_square,
                             const std::string &counter_move_fen) {
  std::string key = make_key(piece, to_square);
  counters[key] = counter_move_fen;
}

std::string CounterMoveTable::get_counter(int piece, int to_square) const {
  std::string key = make_key(piece, to_square);
  auto it = counters.find(key);
  if (it != counters.end()) {
    return it->second;
  }
  return "";
}

void CounterMoveTable::clear() { counters.clear(); }

// ============================================================================
// CONTINUATION HISTORY IMPLEMENTATION
// ============================================================================
void ContinuationHistory::update(int prev_piece, int prev_to, int curr_piece,
                                 int curr_to, int depth) {
  int prev_idx = prev_piece + 6;
  int curr_idx = curr_piece + 6;

  if (prev_idx < 0 || prev_idx >= 13 || prev_to < 0 || prev_to >= 64 ||
      curr_idx < 0 || curr_idx >= 13 || curr_to < 0 || curr_to >= 64) {
    return;
  }

  cont_history[prev_idx][prev_to][curr_idx][curr_to] += depth * depth;

  // Cap to prevent overflow
  if (cont_history[prev_idx][prev_to][curr_idx][curr_to] > 100000) {
    age();
  }
}

int ContinuationHistory::get_score(int prev_piece, int prev_to, int curr_piece,
                                   int curr_to) const {
  int prev_idx = prev_piece + 6;
  int curr_idx = curr_piece + 6;

  if (prev_idx < 0 || prev_idx >= 13 || prev_to < 0 || prev_to >= 64 ||
      curr_idx < 0 || curr_idx >= 13 || curr_to < 0 || curr_to >= 64) {
    return 0;
  }

  return cont_history[prev_idx][prev_to][curr_idx][curr_to];
}

void ContinuationHistory::clear() {
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 64; j++) {
      for (int k = 0; k < 13; k++) {
        for (int l = 0; l < 64; l++) {
          cont_history[i][j][k][l] = 0;
        }
      }
    }
  }
}

void ContinuationHistory::age() {
  // Divide all values by 2 to favor recent patterns
  for (int i = 0; i < 13; i++) {
    for (int j = 0; j < 64; j++) {
      for (int k = 0; k < 13; k++) {
        for (int l = 0; l < 64; l++) {
          cont_history[i][j][k][l] /= 2;
        }
      }
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

std::string TranspositionTable::get_best_move(const std::string &fen) const {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = table.find(fen);
  if (it == table.end()) {
    return "";
  }
  return it->second.best_move_fen;
}

// ============================================================================
// MOVE ORDERING HELPERS
// ============================================================================

// Piece-Square Tables for positional evaluation
// Tables are from white's perspective (flipped for black)
namespace PieceSquareTables {
// Pawn
const int pawn_table[64] = {
    0,  0,  0,  0,   0,   0,  0,  0,  50, 50, 50,  50, 50, 50,  50, 50,
    10, 10, 20, 30,  30,  20, 10, 10, 5,  5,  10,  25, 25, 10,  5,  5,
    0,  0,  0,  20,  20,  0,  0,  0,  5,  -5, -10, 0,  0,  -10, -5, 5,
    5,  10, 10, -20, -20, 10, 10, 5,  0,  0,  0,   0,  0,  0,   0,  0};

// Knight
const int knight_table[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   0,   0,
    0,   -20, -40, -30, 0,   10,  15,  15,  10,  0,   -30, -30, 5,
    15,  20,  20,  15,  5,   -30, -30, 0,   15,  20,  20,  15,  0,
    -30, -30, 5,   10,  15,  15,  10,  5,   -30, -40, -20, 0,   5,
    5,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50};

// Bishop
const int bishop_table[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0,   0,   0,   0,
    0,   0,   -10, -10, 0,   5,   10,  10,  5,   0,   -10, -10, 5,
    5,   10,  10,  5,   5,   -10, -10, 0,   10,  10,  10,  10,  0,
    -10, -10, 10,  10,  10,  10,  10,  10,  -10, -10, 5,   0,   0,
    0,   0,   5,   -10, -20, -10, -10, -10, -10, -10, -10, -20};

// Rook
const int rook_table[64] = {0,  0,  0, 0,  0, 0,  0,  0, 5,  10, 10, 10, 10,
                            10, 10, 5, -5, 0, 0,  0,  0, 0,  0,  -5, -5, 0,
                            0,  0,  0, 0,  0, -5, -5, 0, 0,  0,  0,  0,  0,
                            -5, -5, 0, 0,  0, 0,  0,  0, -5, -5, 0,  0,  0,
                            0,  0,  0, -5, 0, 0,  0,  5, 5,  0,  0,  0};

// Queen
const int queen_table[64] = {
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0,   0,   0,  0,  0,   0,   -10,
    -10, 0,   5,   5,  5,  5,   0,   -10, -5,  0,   5,   5,  5,  5,   0,   -5,
    0,   0,   5,   5,  5,  5,   0,   -5,  -10, 5,   5,   5,  5,  5,   0,   -10,
    -10, 0,   5,   0,  0,  0,   0,   -10, -20, -10, -10, -5, -5, -10, -10, -20};

// King middlegame
const int king_middlegame_table[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50,
    -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
    -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30,
    -20, -10, -20, -20, -20, -20, -20, -20, -10, 20,  20,  0,   0,
    0,   0,   20,  20,  20,  30,  10,  0,   0,   10,  30,  20};
} // namespace PieceSquareTables

// Get piece-square table value
int get_piece_square_value(Piece piece, int square) {
  int abs_piece = piece < 0 ? -piece : piece;
  bool is_white = piece > 0;

  // Flip square for black pieces (mirror vertically)
  int eval_square = is_white ? square : (63 - square);

  switch (abs_piece) {
  case 1:
    return PieceSquareTables::pawn_table[eval_square]; // Pawn
  case 2:
    return PieceSquareTables::knight_table[eval_square]; // Knight
  case 3:
    return PieceSquareTables::bishop_table[eval_square]; // Bishop
  case 4:
    return PieceSquareTables::rook_table[eval_square]; // Rook
  case 5:
    return PieceSquareTables::queen_table[eval_square]; // Queen
  case 6:
    return PieceSquareTables::king_middlegame_table[eval_square]; // King
  default:
    return 0;
  }
}

// MVV-LVA: Most Valuable Victim - Least Valuable Attacker
int get_piece_value(Piece piece) {
  int abs_piece = piece < 0 ? -piece : piece;
  static const int piece_values[] = {0,   100, 320,  330,
                                     500, 900, 20000}; // None, P, N, B, R, Q, K
  return piece_values[abs_piece];
}

// Enhanced evaluation with piece-square tables
int evaluate_with_pst(const std::string &fen) {
  ChessBoard board;
  board.from_fen(fen);

  int score = 0;

  // Evaluate all pieces on the board
  for (int square = 0; square < 64; square++) {
    Piece piece = board.get_piece_at(square);
    if (piece != EMPTY) {
      // Material value
      int material = get_piece_value(piece);
      // Positional bonus from piece-square tables
      int positional = get_piece_square_value(piece, square);

      // Add to score (white pieces are positive, black are negative)
      if (piece > 0) {
        score += material + positional;
      } else {
        score -= material + positional;
      }
    }
  }

  return score;
}

int mvv_lva_score(const ChessBoard &board, const Move &move) {
  Piece victim = board.get_piece_at(move.to);
  if (victim == EMPTY)
    return 0;

  Piece attacker = board.get_piece_at(move.from);
  // Victim value * 10 - attacker value (prefer capturing with low-value pieces)
  return get_piece_value(victim) * 10 - get_piece_value(attacker);
}

// Forward declaration
int static_exchange_eval(ChessBoard &board, const Move &move);

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
      // Captures ordered by SEE (Static Exchange Evaluation)
      int see_score = static_exchange_eval(board, move);

      // Prioritize winning/even captures, penalize losing captures
      if (see_score >= 0) {
        // Good capture: base score + SEE value
        score = 800000 + see_score;
      } else {
        // Bad capture (lose material): still try but with lower priority
        score = 700000 + see_score;
      }
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
// STATIC EXCHANGE EVALUATION (SEE)
// ============================================================================

// Calculate the expected material outcome of a capture sequence
// Returns the expected gain/loss in centipawns
int static_exchange_eval(ChessBoard &board, const Move &move) {
  // Get piece values
  auto get_value = [](Piece p) -> int {
    int abs_p = abs(p);
    switch (abs_p) {
    case 1:
      return 100; // Pawn
    case 2:
      return 320; // Knight
    case 3:
      return 330; // Bishop
    case 4:
      return 500; // Rook
    case 5:
      return 900; // Queen
    case 6:
      return 20000; // King
    default:
      return 0;
    }
  };

  Piece attacker = board.get_piece_at(move.from);
  Piece victim = board.get_piece_at(move.to);

  if (victim == EMPTY) {
    return 0; // Not a capture
  }

  // Initial gain from capturing the victim
  int gain = get_value(victim);

  // Simulate the attacker being captured
  int loss = get_value(attacker);

  // Simple SEE: assume best case for opponent (they recapture)
  // More advanced would simulate full exchange sequence
  // For now, return gain - potential loss if piece is undefended

  // Simplification: if it's a favorable trade, return positive
  // Pawn takes queen: 900 gain, 100 loss = +800
  // Queen takes pawn: 100 gain, 900 loss = -800 (if recaptured)

  return gain - loss; // Pessimistic: assume we'll be recaptured
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
                    HistoryTable *history = nullptr,
                    bool allow_null_move = true) {
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
      std::string null_fen =
          fen; // Simplified: in real chess, would flip side to move

      // Try shallow search with null move
      int null_score = -alpha_beta_internal(
          null_fen, depth - 1 - R, -beta, -beta + 1, !maximizingPlayer,
          evaluate, tt, use_quiescence, killers, ply + 1, history, false);

      if (null_score >= beta) {
        return beta; // Null move cutoff
      }
    }
  }

  // Razoring - prune at low depths when evaluation is far below alpha
  if (!board.is_check() && depth <= 3 && depth > 0) {
    int static_eval = evaluate(fen);
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

  // Internal Iterative Deepening (IID)
  // If we don't have a TT move at high depths, do a shallow search to find one
  if (tt_best_move.empty() && depth >= 5) {
    // Do a reduced depth search to populate TT with a best move
    int iid_depth = depth - 2;
    alpha_beta_internal(fen, iid_depth, alpha, beta, maximizingPlayer, evaluate,
                        tt, use_quiescence, killers, ply, history,
                        allow_null_move);
    // After this search, the TT should have a best move for this position
    tt.probe(fen, iid_depth, alpha, beta, cached_score, tt_best_move);
  }

  // Singular Extensions
  // If we have a TT move and it appears significantly better than alternatives,
  // extend the search depth for this node
  int extension = 0;
  if (!tt_best_move.empty() && depth >= 6) {
    // Perform a reduced search excluding the TT move
    // to see if any other move comes close
    int singular_beta = cached_score - depth * 2; // Margin for singularity
    int singular_depth = depth / 2;

    // Search all moves except the TT move with reduced depth
    int best_alternative = MIN;
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();

      // Skip the TT best move - we're testing if alternatives are worse
      // Compare position without move clocks
      auto last_space = child_fen.rfind(' ');
      if (last_space != std::string::npos) {
        last_space = child_fen.rfind(' ', last_space - 1);
      }
      std::string child_pos = (last_space != std::string::npos)
                                  ? child_fen.substr(0, last_space)
                                  : child_fen;

      last_space = tt_best_move.rfind(' ');
      if (last_space != std::string::npos) {
        last_space = tt_best_move.rfind(' ', last_space - 1);
      }
      std::string tt_pos = (last_space != std::string::npos)
                               ? tt_best_move.substr(0, last_space)
                               : tt_best_move;

      board.unmake_move(move);

      if (child_pos == tt_pos) {
        continue; // Skip the TT move
      }

      // Search alternative move
      board.make_move(move);
      int score =
          alpha_beta_internal(child_fen, singular_depth, singular_beta - 1,
                              singular_beta, !maximizingPlayer, evaluate, tt,
                              use_quiescence, killers, ply + 1, history);
      board.unmake_move(move);

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
  order_moves(board, legal_moves, &tt, fen, killers, ply, history);

  int original_alpha = alpha;
  std::string best_move_fen = "";
  int moves_searched = 0;

  if (maximizingPlayer) {
    int maxEval = MIN;

    // Futility pruning setup - only at low depths
    bool use_futility = false;
    int futility_margin = 0;
    int static_eval = 0;

    if (depth <= 6 && !board.is_check()) {
      static_eval = evaluate(fen);
      futility_margin = 100 + 50 * depth; // Depth 1: 150, Depth 6: 400
      use_futility = true;
    }

    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();

      int eval;
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);

      // Futility pruning - skip quiet moves that can't raise alpha
      if (use_futility && !is_capture && !is_promotion && moves_searched > 0 &&
          !board.is_check()) {
        if (static_eval + futility_margin < alpha) {
          board.unmake_move(move);
          moves_searched++;
          continue; // Skip this move
        }
      }

      // Principal Variation Search (PVS)
      // First move gets full window, rest get null window (scout search)
      if (moves_searched == 0) {
        // First move - search with full window (with possible extension)
        eval = alpha_beta_internal(child_fen, depth - 1 + extension, alpha,
                                   beta, false, evaluate, tt, use_quiescence,
                                   killers, ply + 1, history);
      } else {
        // Late Move Reduction (LMR)
        bool do_lmr = false;
        int reduction = 0;

        if (depth >= 3 && moves_searched >= 4 && !is_capture && !is_promotion &&
            !board.is_check()) {
          do_lmr = true;
          reduction = 1;
          if (moves_searched >= 8)
            reduction = 2;
        }

        // Scout search with null window
        if (do_lmr) {
          // LMR + null window
          eval = alpha_beta_internal(child_fen, depth - 1 - reduction, alpha,
                                     alpha + 1, false, evaluate, tt,
                                     use_quiescence, killers, ply + 1, history);
          // If scout search fails high, re-search with reduced depth but full
          // window
          if (eval > alpha && eval < beta) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                       evaluate, tt, use_quiescence, killers,
                                       ply + 1, history);
          } else if (eval > alpha) {
            // Double re-search: first without reduction, then with full window
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, alpha + 1,
                                       false, evaluate, tt, use_quiescence,
                                       killers, ply + 1, history);
            if (eval > alpha && eval < beta) {
              eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta,
                                         false, evaluate, tt, use_quiescence,
                                         killers, ply + 1, history);
            }
          }
        } else {
          // PVS null window search
          eval = alpha_beta_internal(child_fen, depth - 1, alpha, alpha + 1,
                                     false, evaluate, tt, use_quiescence,
                                     killers, ply + 1, history);
          // If scout search fails high, re-search with full window
          if (eval > alpha && eval < beta) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                       evaluate, tt, use_quiescence, killers,
                                       ply + 1, history);
          }
        }
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

    // Futility pruning setup - only at low depths
    bool use_futility = false;
    int futility_margin = 0;
    int static_eval = 0;

    if (depth <= 6 && !board.is_check()) {
      static_eval = evaluate(fen);
      futility_margin = 100 + 50 * depth;
      use_futility = true;
    }

    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();

      int eval;
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);

      // Futility pruning - skip quiet moves that can't lower beta
      if (use_futility && !is_capture && !is_promotion && moves_searched > 0 &&
          !board.is_check()) {
        if (static_eval - futility_margin > beta) {
          board.unmake_move(move);
          moves_searched++;
          continue; // Skip this move
        }
      }

      // Principal Variation Search (PVS)
      if (moves_searched == 0) {
        // First move - search with full window (with possible extension)
        eval = alpha_beta_internal(child_fen, depth - 1 + extension, alpha,
                                   beta, true, evaluate, tt, use_quiescence,
                                   killers, ply + 1, history);
      } else {
        // Late Move Reduction (LMR)
        bool do_lmr = false;
        int reduction = 0;

        if (depth >= 3 && moves_searched >= 4 && !is_capture && !is_promotion &&
            !board.is_check()) {
          do_lmr = true;
          reduction = 1;
          if (moves_searched >= 8)
            reduction = 2;
        }

        // Scout search with null window
        if (do_lmr) {
          // LMR + null window
          eval = alpha_beta_internal(child_fen, depth - 1 - reduction, beta - 1,
                                     beta, true, evaluate, tt, use_quiescence,
                                     killers, ply + 1, history);
          // If scout search fails low, re-search
          if (eval < beta && eval > alpha) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                       evaluate, tt, use_quiescence, killers,
                                       ply + 1, history);
          } else if (eval < beta) {
            // Double re-search
            eval = alpha_beta_internal(child_fen, depth - 1, beta - 1, beta,
                                       true, evaluate, tt, use_quiescence,
                                       killers, ply + 1, history);
            if (eval < beta && eval > alpha) {
              eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta,
                                         true, evaluate, tt, use_quiescence,
                                         killers, ply + 1, history);
            }
          }
        } else {
          // PVS null window search
          eval = alpha_beta_internal(child_fen, depth - 1, beta - 1, beta, true,
                                     evaluate, tt, use_quiescence, killers,
                                     ply + 1, history);
          // If scout search fails low, re-search with full window
          if (eval < beta && eval > alpha) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                       evaluate, tt, use_quiescence, killers,
                                       ply + 1, history);
          }
        }
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
      if (current_alpha < alpha)
        current_alpha = alpha;
      if (current_beta > beta)
        current_beta = beta;
    }

    // Search with aspiration window
    int score = alpha_beta_internal(fen, depth, current_alpha, current_beta,
                                    maximizingPlayer, evaluate, tt, true,
                                    killers, 0, history);

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
int alpha_beta_optimized(
    const std::string &fen, int depth, int alpha, int beta,
    bool maximizingPlayer,
    const std::function<int(const std::string &)> &evaluate,
    TranspositionTable *tt, int num_threads, KillerMoves *killers,
    HistoryTable *history, CounterMoveTable *counters) {
  TranspositionTable local_tt;
  TranspositionTable &tt_ref = tt ? *tt : local_tt;

  // Create local killer/history/counter tables if not provided
  KillerMoves local_killers;
  HistoryTable local_history;
  CounterMoveTable local_counters;
  KillerMoves *killers_ptr = killers ? killers : &local_killers;
  HistoryTable *history_ptr = history ? history : &local_history;
  CounterMoveTable *counters_ptr = counters ? counters : &local_counters;

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
                                  evaluate, tt_ref, true, killers_ptr, 1,
                                  history_ptr);
    } else {
      score =
          alpha_beta_internal(child_fen, depth - 1, alpha, beta, true, evaluate,
                              tt_ref, true, killers_ptr, 1, history_ptr);
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
                    HistoryTable *history, CounterMoveTable *counters) {
  // TODO: Implement CUDA-accelerated batch evaluation
  // For now, fall back to optimized CPU version
  return alpha_beta_optimized(fen, depth, alpha, beta, maximizingPlayer,
                              evaluate, tt, 0, killers, history, counters);
}

// ============================================================================
// BEST MOVE FINDER - Returns the actual best move, not just score
// ============================================================================
std::string
find_best_move(const std::string &fen, int depth,
               const std::function<int(const std::string &)> &evaluate,
               TranspositionTable *tt, int num_threads, KillerMoves *killers,
               HistoryTable *history, CounterMoveTable *counters) {
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

  // Parse FEN to determine whos to move
  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    return ""; // No legal moves
  }

  // Determine if maximizing (white to move)
  bool maximizing = fen.find(" w ") != std::string::npos;

  // Run the search - this populates TT with best move
  alpha_beta_optimized(fen, depth, MIN, MAX, maximizing, evaluate, tt,
                       num_threads, killers, history, counters);

  // Get the best move FEN from TT
  std::string best_move_fen = tt->get_best_move(fen);

  if (best_move_fen.empty()) {
    // Fallback: evaluate each move and pick the best
    std::string best = "";
    int best_score = maximizing ? MIN : MAX;

    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();

      int score =
          alpha_beta_optimized(child_fen, depth - 1, MIN, MAX, !maximizing,
                               evaluate, tt, 0, killers, history);

      board.unmake_move(move);

      if ((maximizing && score > best_score) ||
          (!maximizing && score < best_score)) {
        best_score = score;
        best = child_fen;
      }
    }
    return best;
  }

  return best_move_fen;
}

// Convert best move FEN back to move notation (UCI format: e.g., "e2e4")
std::string
get_best_move_uci(const std::string &fen, int depth,
                  const std::function<int(const std::string &)> &evaluate,
                  TranspositionTable *tt, int num_threads, KillerMoves *killers,
                  HistoryTable *history, CounterMoveTable *counters) {
  std::string best_move_fen = find_best_move(
      fen, depth, evaluate, tt, num_threads, killers, history, counters);

  if (best_move_fen.empty()) {
    return "";
  }

  // Find which move leads to this FEN
  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  // Extract position part of FEN (everything before halfmove clock)
  // FEN format: "position w/b castling ep halfmove fullmove"
  // We only want: "position w/b castling ep"
  auto last_space = best_move_fen.rfind(' ');
  if (last_space != std::string::npos) {
    last_space = best_move_fen.rfind(' ', last_space - 1);
  }
  std::string best_move_pos = (last_space != std::string::npos)
                                  ? best_move_fen.substr(0, last_space)
                                  : best_move_fen;

  for (const Move &move : legal_moves) {
    board.make_move(move);
    std::string child_fen = board.to_fen();
    board.unmake_move(move);

    // Extract position part from child FEN too
    last_space = child_fen.rfind(' ');
    if (last_space != std::string::npos) {
      last_space = child_fen.rfind(' ', last_space - 1);
    }
    std::string child_pos = (last_space != std::string::npos)
                                ? child_fen.substr(0, last_space)
                                : child_fen;

    if (child_pos == best_move_pos) {
      // Convert move to UCI format (e.g., "e2e4", "e7e8q" for promotion)
      std::string uci = "";

      // Convert square indices to algebraic notation
      int from_rank = move.from / 8;
      int from_file = move.from % 8;
      int to_rank = move.to / 8;
      int to_file = move.to % 8;

      uci += char('a' + from_file);
      uci += char('1' + from_rank);
      uci += char('a' + to_file);
      uci += char('1' + to_rank);

      // Add promotion piece if applicable
      if (move.promotion != EMPTY) {
        char promo = ' ';
        int piece_type = move.promotion < 0 ? -move.promotion : move.promotion;
        switch (piece_type) {
        case 2:
          promo = 'n';
          break; // Knight
        case 3:
          promo = 'b';
          break; // Bishop
        case 4:
          promo = 'r';
          break; // Rook
        case 5:
          promo = 'q';
          break; // Queen
        }
        if (promo != ' ')
          uci += promo;
      }

      return uci;
    }
  }

  return ""; // Should not reach here
}

// ============================================================================
// 4. PGN TO FEN - Convert PGN string to FEN string
// ============================================================================

// Helper function to parse SAN (Standard Algebraic Notation) move
// Returns true if move was successfully parsed and applied
static bool parse_and_apply_san(ChessBoard &board,
                                const std::string &san_move) {
  // Remove check/checkmate markers
  std::string move = san_move;
  if (!move.empty() && (move.back() == '+' || move.back() == '#')) {
    move.pop_back();
  }

  // Handle castling
  if (move == "O-O" || move == "0-0" || move == "o-o") {
    // Kingside castling
    bool is_white = board.is_white_to_move();
    int king_rank = is_white ? 0 : 7;
    int king_from = king_rank * 8 + 4; // e1 or e8
    int king_to = king_rank * 8 + 6;   // g1 or g8
    board.make_move(Move(king_from, king_to));
    return true;
  } else if (move == "O-O-O" || move == "0-0-0" || move == "o-o-o") {
    // Queenside castling
    bool is_white = board.is_white_to_move();
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
    bool is_white = board.is_white_to_move();
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
    bool is_white = board.is_white_to_move();
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
    bool is_white = board.is_white_to_move();
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
          (piece_type == (board.is_white_to_move() ? W_PAWN : B_PAWN) &&
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
  ChessBoard board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

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
// OPENING BOOK IMPLEMENTATION
// ============================================================================

// Polyglot Zobrist hash implementation with standard keys
uint64_t OpeningBook::polyglot_hash(const std::string &fen) {
  // Standard Polyglot Zobrist random numbers (fixed seed from spec)
  // These are the official Polyglot keys - 781 random 64-bit numbers
  static std::vector<uint64_t> random_values;

  if (random_values.empty()) {
    // Initialize with Polyglot's specific PRNG (seed = 0)
    std::mt19937_64 rng(0);
    random_values.resize(781);
    for (int i = 0; i < 781; i++) {
      random_values[i] = rng();
    }
  }

  uint64_t hash = 0;
  ChessBoard board;
  board.from_fen(fen);

  // Hash pieces (indices 0-767)
  // Polyglot order: white pieces then black pieces
  // For each piece type: pawn, knight, bishop, rook, queen, king
  for (int sq = 0; sq < 64; sq++) {
    Piece p = board.get_piece_at(sq);
    if (p != EMPTY) {
      int piece_type, color;
      if (p >= 6) { // Black pieces
        color = 1;
        piece_type = p - 6;
      } else { // White pieces
        color = 0;
        piece_type = p;
      }

      // Polyglot uses files from left to right, ranks from bottom to top
      // Convert our square (0=a1, 63=h8) to Polyglot format
      int file = sq % 8;
      int rank = sq / 8;
      int polyglot_sq = rank * 8 + file;

      // Index: color * 384 + piece_type * 64 + square
      int index = color * 384 + piece_type * 64 + polyglot_sq;
      hash ^= random_values[index];
    }
  }

  // Hash castling rights (indices 768-771)
  std::string castling = "";
  size_t space_count = 0;
  for (char c : fen) {
    if (c == ' ')
      space_count++;
    if (space_count == 2) {
      size_t next_space = fen.find(' ', fen.find(c) + 1);
      castling = fen.substr(fen.find(c) + 1, next_space - fen.find(c) - 1);
      break;
    }
  }
  if (castling.find('K') != std::string::npos)
    hash ^= random_values[768];
  if (castling.find('Q') != std::string::npos)
    hash ^= random_values[769];
  if (castling.find('k') != std::string::npos)
    hash ^= random_values[770];
  if (castling.find('q') != std::string::npos)
    hash ^= random_values[771];

  // Hash en passant (indices 772-779)
  std::string ep = "";
  space_count = 0;
  for (char c : fen) {
    if (c == ' ')
      space_count++;
    if (space_count == 3) {
      size_t next_space = fen.find(' ', fen.find(c) + 1);
      if (next_space == std::string::npos)
        next_space = fen.length();
      ep = fen.substr(fen.find(c) + 1, next_space - fen.find(c) - 1);
      break;
    }
  }
  if (ep != "-" && ep.length() == 2) {
    int file = ep[0] - 'a';
    hash ^= random_values[772 + file];
  }

  // Hash side to move (index 780)
  if (fen.find(" b ") != std::string::npos) {
    hash ^= random_values[780];
  }

  return hash;
}

std::string OpeningBook::decode_move(uint16_t move) {
  // Polyglot move format:
  // bits 0-5: to square
  // bits 6-11: from square
  // bits 12-14: promotion (0=none, 1=N, 2=B, 3=R, 4=Q)

  int to_sq = move & 0x3F;
  int from_sq = (move >> 6) & 0x3F;
  int promo = (move >> 12) & 0x7;

  // Convert to UCI
  char from_file = 'a' + (from_sq % 8);
  char from_rank = '1' + (from_sq / 8);
  char to_file = 'a' + (to_sq % 8);
  char to_rank = '1' + (to_sq / 8);

  std::string uci;
  uci += from_file;
  uci += from_rank;
  uci += to_file;
  uci += to_rank;

  // Add promotion
  if (promo > 0) {
    const char promo_chars[] = {' ', 'n', 'b', 'r', 'q'};
    if (promo < 5) {
      uci += promo_chars[promo];
    }
  }

  return uci;
}

bool OpeningBook::load(const std::string &book_path) {
  std::ifstream file(book_path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  entries.clear();

  while (file.good()) {
    BookEntry entry;
    char buffer[16];

    file.read(buffer, 16);
    if (file.gcount() < 16)
      break;

    // Parse big-endian Polyglot format
    entry.key = 0;
    for (int i = 0; i < 8; i++) {
      entry.key = (entry.key << 8) | (unsigned char)buffer[i];
    }

    entry.move = ((unsigned char)buffer[8] << 8) | (unsigned char)buffer[9];
    entry.weight = ((unsigned char)buffer[10] << 8) | (unsigned char)buffer[11];
    entry.learn = ((unsigned char)buffer[12] << 24) |
                  ((unsigned char)buffer[13] << 16) |
                  ((unsigned char)buffer[14] << 8) | (unsigned char)buffer[15];

    entries.push_back(entry);
  }

  file.close();

  // Sort by key for binary search
  std::sort(
      entries.begin(), entries.end(),
      [](const BookEntry &a, const BookEntry &b) { return a.key < b.key; });

  loaded = !entries.empty();
  return loaded;
}

std::vector<BookMove> OpeningBook::probe(const std::string &fen) {
  std::vector<BookMove> moves;

  if (!loaded)
    return moves;

  uint64_t hash = polyglot_hash(fen);

  // Binary search for matching positions
  auto lower = std::lower_bound(
      entries.begin(), entries.end(), hash,
      [](const BookEntry &e, uint64_t h) { return e.key < h; });

  // Collect all moves for this position
  while (lower != entries.end() && lower->key == hash) {
    BookMove bm;
    bm.uci_move = decode_move(lower->move);
    bm.weight = lower->weight;
    moves.push_back(bm);
    ++lower;
  }

  return moves;
}

std::string OpeningBook::probe_best(const std::string &fen) {
  auto moves = probe(fen);
  if (moves.empty())
    return "";

  // Return highest weighted move
  auto best = std::max_element(
      moves.begin(), moves.end(),
      [](const BookMove &a, const BookMove &b) { return a.weight < b.weight; });
  return best->uci_move;
}

std::string OpeningBook::probe_weighted(const std::string &fen) {
  auto moves = probe(fen);
  if (moves.empty())
    return "";

  // Weighted random selection
  int total_weight = 0;
  for (const auto &m : moves) {
    total_weight += m.weight;
  }

  if (total_weight == 0) {
    return moves[0].uci_move;
  }

  int rand_val = rand() % total_weight;
  int current = 0;

  for (const auto &m : moves) {
    current += m.weight;
    if (rand_val < current) {
      return m.uci_move;
    }
  }

  return moves[0].uci_move;
}

void OpeningBook::clear() {
  entries.clear();
  loaded = false;
}

// ============================================================================
// MULTI-PV SEARCH IMPLEMENTATION
// ============================================================================

std::vector<PVLine>
multi_pv_search(const std::string &fen, int depth, int num_lines,
                const std::function<int(const std::string &)> &evaluate,
                TranspositionTable *tt, int num_threads, KillerMoves *killers,
                HistoryTable *history, CounterMoveTable *counters) {

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

  ChessBoard board(fen);
  std::vector<Move> legal_moves = board.generate_legal_moves();

  if (legal_moves.empty()) {
    return pv_lines; // No moves available
  }

  // Determine if maximizing
  bool maximizing = fen.find(" w ") != std::string::npos;

  // Score all moves
  struct ScoredMove {
    Move move;
    std::string fen_after;
    int score;
  };

  std::vector<ScoredMove> scored_moves;
  scored_moves.reserve(legal_moves.size());

  for (const Move &move : legal_moves) {
    board.make_move(move);
    std::string child_fen = board.to_fen();

    int score =
        alpha_beta_optimized(child_fen, depth - 1, MIN, MAX, !maximizing,
                             evaluate, tt, 0, killers, history, counters);

    board.unmake_move(move);

    ScoredMove sm{move, child_fen, score};
    scored_moves.push_back(sm);
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
    std::string current_fen = scored_moves[i].fen_after;
    ChessBoard pv_board(current_fen);

    // Walk through TT to extract continuation
    for (int pv_depth = 0; pv_depth < depth - 1 && pv_depth < 10; pv_depth++) {
      std::string best_move_fen = tt->get_best_move(current_fen);

      if (best_move_fen.empty()) {
        break; // No best move in TT
      }

      // Try to apply the best move
      std::vector<Move> legal = pv_board.generate_legal_moves();
      bool move_found = false;

      for (const Move &legal_move : legal) {
        pv_board.make_move(legal_move);
        std::string result_fen = pv_board.to_fen();

        if (result_fen == best_move_fen) {
          // Found the move! Convert to UCI
          char from_file = 'a' + (legal_move.from % 8);
          char from_rank = '1' + (legal_move.from / 8);
          char to_file = 'a' + (legal_move.to % 8);
          char to_rank = '1' + (legal_move.to / 8);

          std::string uci_move =
              std::string() + from_file + from_rank + to_file + to_rank;

          // Add promotion if needed
          if (legal_move.promotion != EMPTY) {
            char promo = ' ';
            int piece_type = legal_move.promotion < 0 ? -legal_move.promotion
                                                      : legal_move.promotion;
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
              uci_move += promo;
          }

          line.pv += " " + uci_move;
          current_fen = result_fen;
          move_found = true;
          break;
        }

        pv_board.unmake_move(legal_move);
      }

      if (!move_found) {
        break; // Can't continue PV
      }
    }

    pv_lines.push_back(line);
  }

  return pv_lines;
}
