#include "functions.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(c_helpers, m) {
  m.doc() = "ChessHacks C++ extension module";

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

  // CUDA availability check
  m.def("is_cuda_available", &is_cuda_available,
        "Check if CUDA is available for GPU acceleration.\n"
        "Returns True if CUDA devices are detected and accessible.");

  m.def("get_cuda_info", &get_cuda_info,
        "Get information about available CUDA devices.\n"
        "Returns a string describing the GPU and CUDA version.");

  // 4. PGN to FEN: Convert PGN string to FEN string
  m.def("pgn_to_fen", &pgn_to_fen,
        nb::arg("pgn"),
        "Convert PGN (Portable Game Notation) string to FEN (Forsyth-Edwards Notation) string.\n"
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
}
