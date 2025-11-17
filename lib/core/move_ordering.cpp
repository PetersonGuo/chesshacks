#include "core/move_ordering.h"

// ============================================================================
// KILLER MOVES IMPLEMENTATION
// ============================================================================

void KillerMoves::store(int ply, uint16_t move) {
  if (ply >= MAX_DEPTH)
    return;

  // Don't store duplicates
  if (killers[ply][0] == move)
    return;

  // Shift: second killer becomes first, new move becomes second
  killers[ply][1] = killers[ply][0];
  killers[ply][0] = move;
}

bool KillerMoves::is_killer(int ply, uint16_t move) const {
  if (ply >= MAX_DEPTH)
    return false;
  return killers[ply][0] == move || killers[ply][1] == move;
}

void KillerMoves::clear() {
  for (int i = 0; i < MAX_DEPTH; i++) {
    killers[i][0] = 0;
    killers[i][1] = 0;
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

void CounterMoveTable::store(int piece, int to_square, uint16_t counter_move) {
  int idx = piece + 6;
  if (idx < 0 || idx >= 13 || to_square < 0 || to_square >= 64)
    return;
  counters[idx][to_square] = counter_move;
}

uint16_t CounterMoveTable::get_counter(int piece, int to_square) const {
  int idx = piece + 6;
  if (idx < 0 || idx >= 13 || to_square < 0 || to_square >= 64)
    return 0;
  return counters[idx][to_square];
}

void CounterMoveTable::clear() {
  for (auto &row : counters) {
    row.fill(0);
  }
}

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
