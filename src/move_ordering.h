#ifndef MOVE_ORDERING_H
#define MOVE_ORDERING_H

#include <array>
#include <cstdint>

const int MAX_DEPTH = 64; // Maximum search depth for killer moves

// Killer moves table (stores 2 killer moves per ply)
struct KillerMoves {
  std::array<std::array<uint16_t, 2>, MAX_DEPTH> killers{};

  void store(int ply, uint16_t move);
  bool is_killer(int ply, uint16_t move) const;
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
  void store(int piece, int to_square, uint16_t counter_move);
  uint16_t get_counter(int piece, int to_square) const;
  void clear();

private:
  // [piece_type + 6][to_square] -> counter move (encoded)
  std::array<std::array<uint16_t, 64>, 13> counters{};
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
