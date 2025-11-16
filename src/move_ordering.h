#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include <string>
#include <unordered_map>

const int MAX_DEPTH = 64; // Maximum search depth for killer moves

// Killer moves table (stores 2 killer moves per ply)
struct KillerMoves {
  std::string killers[MAX_DEPTH][2];

  void store(int ply, const std::string &move_fen);
  bool is_killer(int ply, const std::string &move_fen) const;
  void clear();
};

// History heuristic table (piece-to-square)
class HistoryTable {
public:
  void update(int piece, int to_square, int depth);
  int get_score(int piece, int to_square) const;
  void clear();
  void age(); // Age history scores to favor recent moves

private:
  // [piece_type + 6][to_square] - piece_type ranges from -6 to 6
  int history[13][64] = {{0}};
};

// Counter move table (stores refutation moves)
// Tracks which move was best in response to opponent's last move
class CounterMoveTable {
public:
  void store(int piece, int to_square, const std::string &counter_move_fen);
  std::string get_counter(int piece, int to_square) const;
  void clear();

private:
  // [piece_type + 6][to_square] -> counter move FEN
  std::unordered_map<std::string, std::string> counters;

  std::string make_key(int piece, int to_square) const {
    return std::to_string(piece) + "_" + std::to_string(to_square);
  }
};

// Continuation history table (tracks two-move patterns)
// Extends history heuristic with follow-up move sequences
class ContinuationHistory {
public:
  void update(int prev_piece, int prev_to, int curr_piece, int curr_to,
              int depth);
  int get_score(int prev_piece, int prev_to, int curr_piece, int curr_to) const;
  void clear();
  void age();

private:
  // [prev_piece+6][prev_to][curr_piece+6][curr_to] -> score
  int cont_history[13][64][13][64] = {{{{0}}}};
};

#endif // MOVE_ORDERING_H
