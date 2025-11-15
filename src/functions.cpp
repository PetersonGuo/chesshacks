#include "functions.h"
#include "chess_board.h"
#include <algorithm>
#include <vector>

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