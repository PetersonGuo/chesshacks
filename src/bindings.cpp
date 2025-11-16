#include "functions.h"
#ifdef CUDA_ENABLED
#include "cuda/cuda_eval.h"
#include "evaluation.h"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(c_helpers, m) {
  m.doc() = "ChessHacks C++ extension module";

#ifdef CUDA_ENABLED
  // Initialize CUDA piece-square tables
  cuda_init_tables(
      PieceSquareTables::pawn_table, PieceSquareTables::knight_table,
      PieceSquareTables::bishop_table, PieceSquareTables::rook_table,
      PieceSquareTables::queen_table, PieceSquareTables::king_middlegame_table);
#endif

  // Expose constants as attributes
  m.attr("MIN") = MIN;
  m.attr("MAX") = MAX;

  // Expose TranspositionTable class
  nb::class_<TranspositionTable>(m, "TranspositionTable")
      .def(nb::init<>(), "Create a new transposition table")
      .def("clear", &TranspositionTable::clear, "Clear all cached positions")
      .def("size", &TranspositionTable::size, "Get number of cached positions")
      .def("__len__", &TranspositionTable::size);

  // Expose KillerMoves class
  nb::class_<KillerMoves>(m, "KillerMoves")
      .def(nb::init<>(), "Create a new killer moves table")
      .def("clear", &KillerMoves::clear, "Clear all killer moves");

  // Expose HistoryTable class
  nb::class_<HistoryTable>(m, "HistoryTable")
      .def(nb::init<>(), "Create a new history heuristic table")
      .def("clear", &HistoryTable::clear, "Clear all history scores")
      .def("age", &HistoryTable::age, "Age history scores (divide by 2)");

  // Expose CounterMoveTable class
  nb::class_<CounterMoveTable>(m, "CounterMoveTable")
      .def(nb::init<>(), "Create a new counter move table")
      .def("clear", &CounterMoveTable::clear, "Clear all counter moves");

  // 1. BASIC: Bare-bones alpha-beta (no optimizations) - BACKUP
  m.def("alpha_beta_basic", &alpha_beta_basic, nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), nb::arg("maximizingPlayer"),
        nb::arg("evaluate"),
        "Basic alpha-beta pruning search (no optimizations).\n"
        "This is the fallback version with minimal dependencies.\n"
        "evaluate: Python callback function that takes a FEN string and "
        "returns an int score");

  // 2. OPTIMIZED: Full optimizations (TT + move ordering + parallel + killer
  // moves + history + null move pruning)
  m.def("alpha_beta_optimized", &alpha_beta_optimized, nb::arg("fen"),
        nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("evaluate"),
        nb::arg("tt") = nullptr, nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "Optimized alpha-beta with all enhancements:\n"
        "- Transposition table caching\n"
        "- Advanced move ordering (TT + killer moves + MVV-LVA + history + "
        "counter moves)\n"
        "- Quiescence search\n"
        "- Iterative deepening\n"
        "- Null move pruning\n"
        "- Singular extensions\n"
        "- Optional multithreading\n\n"
        "tt: Optional TranspositionTable instance (creates local one if null)\n"
        "num_threads: Number of threads for parallel search (0 = auto, 1 = "
        "sequential)\n"
        "killers: Optional KillerMoves instance (creates local one if null)\n"
        "history: Optional HistoryTable instance (creates local one if null)\n"
        "counters: Optional CounterMoveTable instance (creates local one if "
        "null)");

  // 3. CUDA: GPU-accelerated search (falls back to optimized)
  m.def("alpha_beta_cuda", &alpha_beta_cuda, nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), nb::arg("maximizingPlayer"),
        nb::arg("evaluate"), nb::arg("tt") = nullptr,
        nb::arg("killers") = nullptr, nb::arg("history") = nullptr,
        nb::arg("counters") = nullptr,
        "CUDA-accelerated alpha-beta search (currently falls back to optimized "
        "CPU version).\n"
        "tt: Optional TranspositionTable instance\n"
        "killers: Optional KillerMoves instance\n"
        "history: Optional HistoryTable instance\n"
        "counters: Optional CounterMoveTable instance");

  // Evaluation function with piece-square tables
  m.def("evaluate_with_pst", &evaluate_with_pst, nb::arg("fen"),
        "Enhanced evaluation function using material + piece-square tables.\n"
        "Provides positional bonuses based on piece placement.");

  // Batch evaluation (multithreaded CPU)
  m.def("batch_evaluate_mt", &batch_evaluate_mt, nb::arg("fens"),
        nb::arg("num_threads") = 0,
        "Batch evaluate multiple positions using multithreading.\n"
        "Much more efficient than evaluating positions one at a time.\n\n"
        "fens: List of FEN strings to evaluate\n"
        "num_threads: Number of threads (0 = auto-detect all available cores)\n"
        "Returns: List of evaluation scores");

  // NNUE evaluation functions
  m.def("init_nnue", &init_nnue, nb::arg("model_path"),
        "Initialize NNUE evaluator with a trained model.\n"
        "Load NNUE model from binary file (.bin).\n\n"
        "model_path: Path to binary NNUE model file\n"
        "Returns: True if loaded successfully, False otherwise");

  m.def("is_nnue_loaded", &is_nnue_loaded,
        "Check if NNUE model is loaded and ready for evaluation.\n"
        "Returns: True if NNUE is loaded, False otherwise");

  m.def("evaluate_nnue",
        nb::overload_cast<const std::string&>(&evaluate_nnue),
        nb::arg("fen"),
        "Evaluate position using NNUE model.\n"
        "Falls back to PST evaluation if NNUE not loaded.\n\n"
        "fen: Position in FEN notation\n"
        "Returns: Evaluation score in centipawns");

  m.def("evaluate",
        nb::overload_cast<const std::string&>(&evaluate),
        nb::arg("fen"),
        "Evaluate position using best available method.\n"
        "Uses NNUE if loaded, otherwise PST evaluation.\n\n"
        "fen: Position in FEN notation\n"
        "Returns: Evaluation score in centipawns");

  // CUDA availability check
  m.def("is_cuda_available", &is_cuda_available,
        "Check if CUDA is available for GPU acceleration.\n"
        "Returns True if CUDA devices are detected and accessible.");

  m.def("get_cuda_info", &get_cuda_info,
        "Get information about available CUDA devices.\n"
        "Returns a string describing the GPU and CUDA version.");

#ifdef CUDA_ENABLED
  // CUDA batch operations - using Python-friendly wrappers
  m.def("cuda_batch_evaluate", &cuda_batch_evaluate_py, nb::arg("fens"),
        "Batch evaluate multiple chess positions on GPU.\n"
        "Much more efficient than evaluating positions one at a time.\n\n"
        "fens: List of FEN strings to evaluate\n"
        "Returns: List of evaluation scores");

  m.def("cuda_batch_count_pieces", &cuda_batch_count_pieces_py, nb::arg("fens"),
        "Count pieces in multiple positions on GPU.\n"
        "Returns counts for each piece type (6 white, 6 black).\n\n"
        "fens: List of FEN strings\n"
        "Returns: List of lists with piece counts");

  m.def("cuda_batch_hash_positions", &cuda_batch_hash_positions_py,
        nb::arg("fens"),
        "Generate hash values for multiple positions on GPU.\n"
        "Used for transposition table lookups.\n\n"
        "fens: List of FEN strings\n"
        "Returns: List of hash values");
#endif

  // 4. PGN to FEN: Convert PGN string to FEN string
  m.def("pgn_to_fen", &pgn_to_fen, nb::arg("pgn"),
        "Convert PGN (Portable Game Notation) string to FEN (Forsyth-Edwards "
        "Notation) string.\n"
        "Parses PGN moves and returns the final position as FEN.\n"
        "pgn: PGN string containing game moves");

  // Best move finders
  m.def("find_best_move", &find_best_move, nb::arg("fen"), nb::arg("depth"),
        nb::arg("evaluate"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Find the best move for a position.\n"
        "Returns the FEN string of the position after the best move.\n\n"
        "fen: Position in FEN notation\n"
        "depth: Search depth\n"
        "evaluate: Evaluation function\n"
        "tt: Optional TranspositionTable\n"
        "num_threads: Number of threads (0=auto)\n"
        "killers: Optional KillerMoves\n"
        "history: Optional HistoryTable\n"
        "counters: Optional CounterMoveTable");

  m.def("get_best_move_uci", &get_best_move_uci, nb::arg("fen"),
        nb::arg("depth"), nb::arg("evaluate"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Find the best move and return it in UCI format (e.g., 'e2e4').\n"
        "Returns the move in UCI notation.\n\n"
        "fen: Position in FEN notation\n"
        "depth: Search depth\n"
        "evaluate: Evaluation function\n"
        "tt: Optional TranspositionTable\n"
        "num_threads: Number of threads (0=auto)\n"
        "killers: Optional KillerMoves\n"
        "history: Optional HistoryTable\n"
        "counters: Optional CounterMoveTable");

  // Opening Book
  nb::class_<BookMove>(m, "BookMove")
      .def_rw("uci_move", &BookMove::uci_move, "UCI move string")
      .def_rw("weight", &BookMove::weight, "Move weight/frequency");

  nb::class_<OpeningBook>(m, "OpeningBook")
      .def(nb::init<>(), "Create a new opening book")
      .def("load", &OpeningBook::load, nb::arg("book_path"),
           "Load a Polyglot opening book file (.bin)")
      .def("is_loaded", &OpeningBook::is_loaded, "Check if book is loaded")
      .def("probe", &OpeningBook::probe, nb::arg("fen"),
           "Probe book for position, returns list of BookMove")
      .def("probe_best", &OpeningBook::probe_best, nb::arg("fen"),
           "Get highest-weighted move for position")
      .def("probe_weighted", &OpeningBook::probe_weighted, nb::arg("fen"),
           "Get random weighted move for position")
      .def("clear", &OpeningBook::clear, "Clear loaded book");

  // Multi-PV Search
  nb::class_<PVLine>(m, "PVLine")
      .def_rw("uci_move", &PVLine::uci_move, "Best move in UCI format")
      .def_rw("score", &PVLine::score, "Evaluation score")
      .def_rw("depth", &PVLine::depth, "Search depth")
      .def_rw("pv", &PVLine::pv, "Principal variation");

  m.def("multi_pv_search", &multi_pv_search, nb::arg("fen"), nb::arg("depth"),
        nb::arg("num_lines"), nb::arg("evaluate"), nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0, nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr, nb::arg("counters") = nullptr,
        "Search for multiple principal variations.\n"
        "Returns list of top N moves with scores.\n\n"
        "fen: Position in FEN notation\n"
        "depth: Search depth\n"
        "num_lines: Number of lines to return\n"
        "evaluate: Evaluation function\n"
        "tt: Optional TranspositionTable\n"
        "num_threads: Number of threads (0=auto)\n"
        "killers: Optional KillerMoves\n"
        "history: Optional HistoryTable\n"
        "counters: Optional CounterMoveTable");
}
