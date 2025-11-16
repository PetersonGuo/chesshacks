#include "evaluation.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <thread>
#include <vector>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

// ============================================================================
// PIECE-SQUARE TABLES
// ============================================================================

// Tables are from white's perspective (flipped for black)
namespace PieceSquareTables {

const int pawn_table[64] = {
    0,  0,  0,  0,   0,   0,  0,  0,  50, 50, 50,  50, 50, 50,  50, 50,
    10, 10, 20, 30,  30,  20, 10, 10, 5,  5,  10,  25, 25, 10,  5,  5,
    0,  0,  0,  20,  20,  0,  0,  0,  5,  -5, -10, 0,  0,  -10, -5, 5,
    5,  10, 10, -20, -20, 10, 10, 5,  0,  0,  0,   0,  0,  0,   0,  0};

const int knight_table[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   0,   0,
    0,   -20, -40, -30, 0,   10,  15,  15,  10,  0,   -30, -30, 5,
    15,  20,  20,  15,  5,   -30, -30, 0,   15,  20,  20,  15,  0,
    -30, -30, 5,   10,  15,  15,  10,  5,   -30, -40, -20, 0,   5,
    5,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50};

const int bishop_table[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0,   0,   0,   0,
    0,   0,   -10, -10, 0,   5,   10,  10,  5,   0,   -10, -10, 5,
    5,   10,  10,  5,   5,   -10, -10, 0,   10,  10,  10,  10,  0,
    -10, -10, 10,  10,  10,  10,  10,  10,  -10, -10, 5,   0,   0,
    0,   0,   5,   -10, -20, -10, -10, -10, -10, -10, -10, -20};

const int rook_table[64] = {0,  0,  0, 0,  0, 0,  0,  0, 5,  10, 10, 10, 10,
                            10, 10, 5, -5, 0, 0,  0,  0, 0,  0,  -5, -5, 0,
                            0,  0,  0, 0,  0, -5, -5, 0, 0,  0,  0,  0,  0,
                            -5, -5, 0, 0,  0, 0,  0,  0, -5, -5, 0,  0,  0,
                            0,  0,  0, -5, 0, 0,  0,  5, 5,  0,  0,  0};

const int queen_table[64] = {
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0,   0,   0,  0,  0,   0,   -10,
    -10, 0,   5,   5,  5,  5,   0,   -10, -5,  0,   5,   5,  5,  5,   0,   -5,
    0,   0,   5,   5,  5,  5,   0,   -5,  -10, 5,   5,   5,  5,  5,   0,   -10,
    -10, 0,   5,   0,  0,  0,   0,   -10, -20, -10, -10, -5, -5, -10, -10, -20};

const int king_middlegame_table[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50,
    -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
    -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30,
    -20, -10, -20, -20, -20, -20, -20, -20, -10, 20,  20,  0,   0,
    0,   0,   20,  20,  20,  30,  10,  0,   0,   10,  30,  20};

} // namespace PieceSquareTables

// ============================================================================
// EVALUATION FUNCTIONS
// ============================================================================

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

int get_piece_value(Piece piece) {
  int abs_piece = piece < 0 ? -piece : piece;
  static const int piece_values[] = {0,   100, 320,  330,
                                     500, 900, 20000}; // None, P, N, B, R, Q, K
  return piece_values[abs_piece];
}

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

// ============================================================================
// STATIC EXCHANGE EVALUATION (SEE)
// ============================================================================

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
// BATCH EVALUATION (Multithreaded)
// ============================================================================

std::vector<int> batch_evaluate_mt(const std::vector<std::string> &fens,
                                    int num_threads) {
  std::vector<int> scores(fens.size());
  
  if (fens.empty()) {
    return scores;
  }
  
  // Auto-detect threads
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4;
  }
  
  // For small batches, use single thread
  if (fens.size() < static_cast<size_t>(num_threads) || num_threads == 1) {
    for (size_t i = 0; i < fens.size(); i++) {
      scores[i] = evaluate_with_pst(fens[i]);
    }
    return scores;
  }
  
  // Divide work among threads
  std::vector<std::thread> threads;
  size_t chunk_size = (fens.size() + num_threads - 1) / num_threads;
  
  auto evaluate_chunk = [&](size_t start, size_t end) {
    for (size_t i = start; i < end && i < fens.size(); i++) {
      scores[i] = evaluate_with_pst(fens[i]);
    }
  };
  
  for (int t = 0; t < num_threads; t++) {
    size_t start = t * chunk_size;
    size_t end = std::min(start + chunk_size, fens.size());
    if (start < fens.size()) {
      threads.emplace_back(evaluate_chunk, start, end);
    }
  }
  
  for (auto &thread : threads) {
    thread.join();
  }
  
  return scores;
}

// ============================================================================
// MOVE ORDERING
// ============================================================================

void order_moves(ChessBoard &board, std::vector<Move> &moves,
                 TranspositionTable *tt, const std::string &fen,
                 KillerMoves *killers, int ply, HistoryTable *history,
                 CounterMoveTable *counters, int prev_piece, int prev_to,
                 ContinuationHistory *cont_history) {
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

    Piece moving_piece = board.get_piece_at(move.from);
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
    } else if (counters && prev_to >= 0) {
      std::string counter = counters->get_counter(prev_piece, prev_to);
      if (!counter.empty() && counter == move_fen) {
        score = 830000;
      }
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
        score = history->get_score(moving_piece, move.to);
      } else {
        // Fallback to center control heuristic
        int to_rank = move.to / 8;
        int to_file = move.to % 8;
        double center_dist = std::abs(to_rank - 3.5) + std::abs(to_file - 3.5);
        score = 10000 - static_cast<int>(center_dist * 1000);
      }
      if (cont_history && prev_to >= 0) {
        score += cont_history->get_score(prev_piece, prev_to, moving_piece,
                                         move.to);
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
