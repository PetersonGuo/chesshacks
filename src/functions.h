#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "chess_board.h"
#include <climits>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

const int MIN = INT_MIN;
const int MAX = INT_MAX;
const int MAX_DEPTH = 64; // Maximum search depth for killer moves

// Transposition table entry types
enum TTEntryType {
  EXACT,       // Exact score
  LOWER_BOUND, // Alpha cutoff (score >= stored value)
  UPPER_BOUND  // Beta cutoff (score <= stored value)
};

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

struct TTEntry {
  int depth;
  int score;
  TTEntryType type;
  std::string best_move_fen; // Store best move for move ordering
};

// Thread-safe transposition table class
class TranspositionTable {
public:
  void store(const std::string &fen, int depth, int score, TTEntryType type,
             const std::string &best_move_fen = "");
  bool probe(const std::string &fen, int depth, int alpha, int beta, int &score,
             std::string &best_move_fen);
  void clear();
  size_t size() const;
  std::string get_best_move(const std::string &fen) const;

private:
  std::unordered_map<std::string, TTEntry> table;
  mutable std::mutex mutex;
};

// 1. BASIC: Bare-bones alpha-beta (no optimizations) - BACKUP
int alpha_beta_basic(const std::string &fen, int depth, int alpha, int beta,
                     bool maximizingPlayer,
                     const std::function<int(const std::string &)> &evaluate);

// 2. OPTIMIZED: Full optimizations (TT + move ordering + parallel)
int alpha_beta_optimized(
    const std::string &fen, int depth, int alpha, int beta,
    bool maximizingPlayer,
    const std::function<int(const std::string &)> &evaluate,
    TranspositionTable *tt = nullptr, int num_threads = 0,
    KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
    CounterMoveTable *counters = nullptr);

// 3. CUDA: GPU-accelerated search (falls back to optimized if CUDA unavailable)
int alpha_beta_cuda(const std::string &fen, int depth, int alpha, int beta,
                    bool maximizingPlayer,
                    const std::function<int(const std::string &)> &evaluate,
                    TranspositionTable *tt = nullptr,
                    KillerMoves *killers = nullptr,
                    HistoryTable *history = nullptr,
                    CounterMoveTable *counters = nullptr);

// HELPER FUNCTIONS
int get_piece_square_value(Piece piece, int square);
int evaluate_with_pst(const std::string &fen);
bool is_cuda_available();
std::string get_cuda_info();

// BEST MOVE FINDERS
std::string
find_best_move(const std::string &fen, int depth,
               const std::function<int(const std::string &)> &evaluate,
               TranspositionTable *tt = nullptr, int num_threads = 0,
               KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
               CounterMoveTable *counters = nullptr);

std::string
get_best_move_uci(const std::string &fen, int depth,
                  const std::function<int(const std::string &)> &evaluate,
                  TranspositionTable *tt = nullptr, int num_threads = 0,
                  KillerMoves *killers = nullptr,
                  HistoryTable *history = nullptr,
                  CounterMoveTable *counters = nullptr);

// 4. PGN to FEN: Convert PGN string to FEN string
std::string pgn_to_fen(const std::string &pgn);

// ============================================================================
// OPENING BOOK
// ============================================================================

struct BookEntry {
  uint64_t key;    // Polyglot hash of position
  uint16_t move;   // Move in Polyglot format
  uint16_t weight; // Weight/frequency
  uint32_t learn;  // Learning data (unused)
};

struct BookMove {
  std::string uci_move;
  int weight;
};

class OpeningBook {
public:
  OpeningBook() : loaded(false) {}
  
  bool load(const std::string &book_path);
  bool is_loaded() const { return loaded; }
  std::vector<BookMove> probe(const std::string &fen);
  std::string probe_best(const std::string &fen);
  std::string probe_weighted(const std::string &fen); // Random weighted selection
  void clear();
  
private:
  std::vector<BookEntry> entries;
  bool loaded;
  
  uint64_t polyglot_hash(const std::string &fen);
  std::string decode_move(uint16_t move);
};

// ============================================================================
// MULTI-PV SEARCH
// ============================================================================

struct PVLine {
  std::string uci_move;  // Best move in UCI format
  int score;             // Evaluation score
  int depth;             // Search depth
  std::string pv;        // Principal variation (sequence of moves)
};

// Search for multiple principal variations
std::vector<PVLine> 
multi_pv_search(const std::string &fen, int depth, int num_lines,
                const std::function<int(const std::string &)> &evaluate,
                TranspositionTable *tt = nullptr, int num_threads = 0,
                KillerMoves *killers = nullptr, HistoryTable *history = nullptr,
                CounterMoveTable *counters = nullptr);

// ============================================================================
// ENDGAME TABLEBASE (Placeholder for Syzygy integration)
// ============================================================================

enum WDLScore {
  TB_LOSS = 0,
  TB_BLESSED_LOSS = 1,
  TB_DRAW = 2,
  TB_CURSED_WIN = 3,
  TB_WIN = 4,
  TB_FAILED = 5  // Probe failed or position not in TB
};

struct TablebaseResult {
  WDLScore wdl;
  int dtz;  // Distance to zeroing move (50-move rule)
  bool success;
};

class Tablebase {
public:
  Tablebase() : initialized(false) {}
  
  bool init(const std::string &path);
  bool is_initialized() const { return initialized; }
  TablebaseResult probe_wdl(const std::string &fen);
  TablebaseResult probe_dtz(const std::string &fen);
  int max_pieces() const { return max_pieces_; }
  
private:
  bool initialized;
  int max_pieces_;
  std::string tb_path;
  
  // TODO: Integrate with Fathom or python-chess-syzygy
  // For now, placeholder that returns TB_FAILED
};

#endif // FUNCTIONS_H
