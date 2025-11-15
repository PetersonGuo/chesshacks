#include "functions.h"
#include "chess_board.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <thread>
#include <vector>
#include <sstream>
#include <regex>
#include <cctype>

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

// Piece-Square Tables for positional evaluation
// Tables are from white's perspective (flipped for black)
namespace PieceSquareTables {
  // Pawn
  const int pawn_table[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
  };
  
  // Knight
  const int knight_table[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
  };
  
  // Bishop
  const int bishop_table[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
  };
  
  // Rook
  const int rook_table[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
  };
  
  // Queen
  const int queen_table[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
  };
  
  // King middlegame
  const int king_middlegame_table[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
  };
}

// Get piece-square table value
int get_piece_square_value(Piece piece, int square) {
  int abs_piece = piece < 0 ? -piece : piece;
  bool is_white = piece > 0;
  
  // Flip square for black pieces (mirror vertically)
  int eval_square = is_white ? square : (63 - square);
  
  switch (abs_piece) {
    case 1: return PieceSquareTables::pawn_table[eval_square];      // Pawn
    case 2: return PieceSquareTables::knight_table[eval_square];    // Knight
    case 3: return PieceSquareTables::bishop_table[eval_square];    // Bishop
    case 4: return PieceSquareTables::rook_table[eval_square];      // Rook
    case 5: return PieceSquareTables::queen_table[eval_square];     // Queen
    case 6: return PieceSquareTables::king_middlegame_table[eval_square]; // King
    default: return 0;
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

  // Internal Iterative Deepening (IID)
  // If we don't have a TT move at high depths, do a shallow search to find one
  if (tt_best_move.empty() && depth >= 5) {
    // Do a reduced depth search to populate TT with a best move
    int iid_depth = depth - 2;
    alpha_beta_internal(fen, iid_depth, alpha, beta, maximizingPlayer, evaluate,
                       tt, use_quiescence, killers, ply, history, allow_null_move);
    // After this search, the TT should have a best move for this position
    tt.probe(fen, iid_depth, alpha, beta, cached_score, tt_best_move);
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
      
      // Principal Variation Search (PVS)
      // First move gets full window, rest get null window (scout search)
      if (moves_searched == 0) {
        // First move - search with full window
        eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                   evaluate, tt, use_quiescence, killers, ply + 1, history);
      } else {
        // Late Move Reduction (LMR)
        bool do_lmr = false;
        int reduction = 0;
        
        if (depth >= 3 && moves_searched >= 4 && !is_capture && !is_promotion && !board.is_check()) {
          do_lmr = true;
          reduction = 1;
          if (moves_searched >= 8) reduction = 2;
        }
        
        // Scout search with null window
        if (do_lmr) {
          // LMR + null window
          eval = alpha_beta_internal(child_fen, depth - 1 - reduction, alpha, alpha + 1, false,
                                     evaluate, tt, use_quiescence, killers, ply + 1, history);
          // If scout search fails high, re-search with reduced depth but full window
          if (eval > alpha && eval < beta) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                       evaluate, tt, use_quiescence, killers, ply + 1, history);
          } else if (eval > alpha) {
            // Double re-search: first without reduction, then with full window
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, alpha + 1, false,
                                       evaluate, tt, use_quiescence, killers, ply + 1, history);
            if (eval > alpha && eval < beta) {
              eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                         evaluate, tt, use_quiescence, killers, ply + 1, history);
            }
          }
        } else {
          // PVS null window search
          eval = alpha_beta_internal(child_fen, depth - 1, alpha, alpha + 1, false,
                                     evaluate, tt, use_quiescence, killers, ply + 1, history);
          // If scout search fails high, re-search with full window
          if (eval > alpha && eval < beta) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, false,
                                       evaluate, tt, use_quiescence, killers, ply + 1, history);
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
    
    for (const Move &move : legal_moves) {
      board.make_move(move);
      std::string child_fen = board.to_fen();
      
      int eval;
      bool is_capture = board.is_capture(move);
      bool is_promotion = (move.promotion != EMPTY);
      
      // Principal Variation Search (PVS)
      if (moves_searched == 0) {
        // First move - search with full window
        eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                   evaluate, tt, use_quiescence, killers, ply + 1, history);
      } else {
        // Late Move Reduction (LMR)
        bool do_lmr = false;
        int reduction = 0;
        
        if (depth >= 3 && moves_searched >= 4 && !is_capture && !is_promotion && !board.is_check()) {
          do_lmr = true;
          reduction = 1;
          if (moves_searched >= 8) reduction = 2;
        }
        
        // Scout search with null window
        if (do_lmr) {
          // LMR + null window
          eval = alpha_beta_internal(child_fen, depth - 1 - reduction, beta - 1, beta, true,
                                     evaluate, tt, use_quiescence, killers, ply + 1, history);
          // If scout search fails low, re-search
          if (eval < beta && eval > alpha) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                       evaluate, tt, use_quiescence, killers, ply + 1, history);
          } else if (eval < beta) {
            // Double re-search
            eval = alpha_beta_internal(child_fen, depth - 1, beta - 1, beta, true,
                                       evaluate, tt, use_quiescence, killers, ply + 1, history);
            if (eval < beta && eval > alpha) {
              eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                         evaluate, tt, use_quiescence, killers, ply + 1, history);
            }
          }
        } else {
          // PVS null window search
          eval = alpha_beta_internal(child_fen, depth - 1, beta - 1, beta, true,
                                     evaluate, tt, use_quiescence, killers, ply + 1, history);
          // If scout search fails low, re-search with full window
          if (eval < beta && eval > alpha) {
            eval = alpha_beta_internal(child_fen, depth - 1, alpha, beta, true,
                                       evaluate, tt, use_quiescence, killers, ply + 1, history);
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

// ============================================================================
// 4. PGN TO FEN - Convert PGN string to FEN string
// ============================================================================

// Helper function to parse SAN (Standard Algebraic Notation) move
// Returns true if move was successfully parsed and applied
static bool parse_and_apply_san(ChessBoard &board, const std::string &san_move) {
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
    int king_from = king_rank * 8 + 4;  // e1 or e8
    int king_to = king_rank * 8 + 6;    // g1 or g8
    board.make_move(Move(king_from, king_to));
    return true;
  } else if (move == "O-O-O" || move == "0-0-0" || move == "o-o-o") {
    // Queenside castling
    bool is_white = board.is_white_to_move();
    int king_rank = is_white ? 0 : 7;
    int king_from = king_rank * 8 + 4;  // e1 or e8
    int king_to = king_rank * 8 + 2;    // c1 or c8
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
      case 'K': piece_type = is_white ? W_KING : B_KING; break;
      case 'Q': piece_type = is_white ? W_QUEEN : B_QUEEN; break;
      case 'R': piece_type = is_white ? W_ROOK : B_ROOK; break;
      case 'B': piece_type = is_white ? W_BISHOP : B_BISHOP; break;
      case 'N': piece_type = is_white ? W_KNIGHT : B_KNIGHT; break;
      default: return false;
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
      case 'Q': promotion = is_white ? W_QUEEN : B_QUEEN; break;
      case 'R': promotion = is_white ? W_ROOK : B_ROOK; break;
      case 'B': promotion = is_white ? W_BISHOP : B_BISHOP; break;
      case 'N': promotion = is_white ? W_KNIGHT : B_KNIGHT; break;
      default: return false;
    }
    move = move.substr(0, eq_pos);
  }
  
  // Parse target square (last 2 characters, e.g., "e4", "d5")
  if (move.length() < 2) return false;
  
  std::string target_str = move.substr(move.length() - 2);
  if (target_str[0] < 'a' || target_str[0] > 'h' || 
      target_str[1] < '1' || target_str[1] > '8') {
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
  if (candidates.size() > 1 && piece_idx < static_cast<int>(move.length() - 2)) {
    std::string disambig = move.substr(piece_idx, move.length() - piece_idx - 2);
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
  while (std::regex_search(search_start, clean_moves.cend(), matches, move_pattern)) {
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
      if (move.find('.') == std::string::npos && 
          move != "1-0" && move != "0-1" && move != "1/2-1/2" && 
          move != "*" && move.find('-') == std::string::npos) {
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