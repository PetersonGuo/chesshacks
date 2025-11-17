#include "core/evaluation.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

const int MIN = INT_MIN;
const int MAX = INT_MAX;

// Global NNUE evaluator instance
std::unique_ptr<NNUEEvaluator> g_nnue_evaluator = nullptr;

int get_piece_value(BoardPiece piece) {
  int abs_piece = piece < 0 ? -piece : piece;
  static const int piece_values[] = {0,   100, 320,  330,
                                     500, 900, 20000}; // None, P, N, B, R, Q, K
  return piece_values[abs_piece];
}

int evaluate_material(const bitboard::BitboardState &board) {
  int score = 0;
  for (int square = 0; square < 64; ++square) {
    BoardPiece piece = board.get_piece_at(square);
    switch (piece) {
    case W_PAWN:
      score += 100;
      break;
    case W_KNIGHT:
      score += 320;
      break;
    case W_BISHOP:
      score += 330;
      break;
    case W_ROOK:
      score += 500;
      break;
    case W_QUEEN:
      score += 900;
      break;
    case B_PAWN:
      score -= 100;
      break;
    case B_KNIGHT:
      score -= 320;
      break;
    case B_BISHOP:
      score -= 330;
      break;
    case B_ROOK:
      score -= 500;
      break;
    case B_QUEEN:
      score -= 900;
      break;
    default:
      break;
    }
  }
  return score;
}

int mvv_lva_score(const bitboard::BitboardState &board, const BoardMove &move) {
  BoardPiece victim = board.get_piece_at(move.to);
  if (victim == EMPTY)
    return 0;

  BoardPiece attacker = board.get_piece_at(move.from);
  // Victim value * 10 - attacker value (prefer capturing with low-value pieces)
  return get_piece_value(victim) * 10 - get_piece_value(attacker);
}

// ============================================================================
// STATIC EXCHANGE EVALUATION (SEE)
// ============================================================================

int static_exchange_eval(bitboard::BitboardState &board,
                         const BoardMove &move) {
  // Get piece values
  auto get_value = [](BoardPiece p) -> int {
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

  BoardPiece attacker = board.get_piece_at(move.from);
  BoardPiece victim = board.get_piece_at(move.to);

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

std::vector<int>
batch_evaluate_mt(const std::vector<bitboard::BitboardState> &boards,
                  int num_threads) {
  std::vector<int> scores(boards.size());

  if (boards.empty()) {
    return scores;
  }

  // Auto-detect threads
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
      num_threads = 4;
  }

  // For small batches, use single thread
  if (boards.size() < static_cast<size_t>(num_threads) || num_threads == 1) {
    for (size_t i = 0; i < boards.size(); i++) {
      scores[i] = evaluate(boards[i]);
    }
    return scores;
  }

  // Divide work among threads
  std::vector<std::thread> threads;
  size_t chunk_size = (boards.size() + num_threads - 1) / num_threads;

  auto evaluate_chunk = [&](size_t start, size_t end) {
    for (size_t i = start; i < end && i < boards.size(); i++) {
      scores[i] = evaluate(boards[i]);
    }
  };

  for (int t = 0; t < num_threads; t++) {
    size_t start = t * chunk_size;
    size_t end = std::min(start + chunk_size, boards.size());
    if (start < boards.size()) {
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

void order_moves(bitboard::BitboardState &board, std::vector<BoardMove> &moves,
                 uint16_t tt_best_move, KillerMoves *killers, int ply,
                 HistoryTable *history, CounterMoveTable *counters,
                 int prev_piece, int prev_to,
                 ContinuationHistory *cont_history) {
  std::vector<std::pair<BoardMove, int>> scored_moves;
  scored_moves.reserve(moves.size());

  for (const BoardMove &move : moves) {
    int score = 0;

    BoardPiece moving_piece = board.get_piece_at(move.from);
    if (tt_best_move != 0 && move.encoded == tt_best_move) {
      score = 1000000; // TT move gets highest priority
    } else if (move.promotion != EMPTY) {
      // Promotions (prioritize queen promotions)
      score = 900000 + get_piece_value(move.promotion);
    } else if (killers && killers->is_killer(ply, move.encoded)) {
      // Killer moves (good quiet moves from sibling nodes)
      score = 850000;
    } else if (counters && prev_to >= 0) {
      uint16_t counter = counters->get_counter(prev_piece, prev_to);
      if (counter != 0 && counter == move.encoded) {
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
        score +=
            cont_history->get_score(prev_piece, prev_to, moving_piece, move.to);
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
// NNUE EVALUATION
// ============================================================================

bool init_nnue(const std::string &model_path) {
  try {
    g_nnue_evaluator = std::make_unique<NNUEEvaluator>();
    bool success = g_nnue_evaluator->load_model(model_path);

    if (!success) {
      g_nnue_evaluator.reset();
      return false;
    }

    std::cout << "NNUE evaluator initialized successfully" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error initializing NNUE: " << e.what() << std::endl;
    g_nnue_evaluator.reset();
    return false;
  }
}

bool is_nnue_loaded() {
  return g_nnue_evaluator != nullptr && g_nnue_evaluator->is_loaded();
}

int evaluate_nnue(const bitboard::BitboardState &board) {
  if (!is_nnue_loaded()) {
    std::cerr << "Warning: NNUE not loaded, falling back to material evaluation"
              << std::endl;
    return evaluate_material(board);
  }

  return g_nnue_evaluator->evaluate(board);
}

int evaluate(const bitboard::BitboardState &board) {
  if (is_nnue_loaded()) {
    return evaluate_nnue(board);
  }
  return evaluate_material(board);
}
