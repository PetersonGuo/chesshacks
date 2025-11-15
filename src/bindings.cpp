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

  // 1. BASIC: Bare-bones alpha-beta (no optimizations) - BACKUP
  m.def("alpha_beta_basic", &alpha_beta_basic, 
        nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), 
        nb::arg("maximizingPlayer"),
        nb::arg("evaluate"),
        "Basic alpha-beta pruning search (no optimizations).\n"
        "This is the fallback version with minimal dependencies.\n"
        "evaluate: Python callback function that takes a FEN string and returns an int score");

  // 2. OPTIMIZED: Full optimizations (TT + move ordering + parallel + killer moves + history + null move pruning)
  m.def("alpha_beta_optimized", &alpha_beta_optimized, 
        nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), 
        nb::arg("maximizingPlayer"),
        nb::arg("evaluate"),
        nb::arg("tt") = nullptr,
        nb::arg("num_threads") = 0,
        nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr,
        "Optimized alpha-beta with all enhancements:\n"
        "- Transposition table caching\n"
        "- Advanced move ordering (TT + killer moves + MVV-LVA + history)\n"
        "- Quiescence search\n"
        "- Iterative deepening\n"
        "- Null move pruning\n"
        "- Optional multithreading\n\n"
        "tt: Optional TranspositionTable instance (creates local one if null)\n"
        "num_threads: Number of threads for parallel search (0 = auto, 1 = sequential)\n"
        "killers: Optional KillerMoves instance (creates local one if null)\n"
        "history: Optional HistoryTable instance (creates local one if null)");

  // 3. CUDA: GPU-accelerated search (falls back to optimized)
  m.def("alpha_beta_cuda", &alpha_beta_cuda, 
        nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), 
        nb::arg("maximizingPlayer"),
        nb::arg("evaluate"),
        nb::arg("tt") = nullptr,
        nb::arg("killers") = nullptr,
        nb::arg("history") = nullptr,
        "CUDA-accelerated alpha-beta search (currently falls back to optimized CPU version).\n"
        "tt: Optional TranspositionTable instance\n"
        "killers: Optional KillerMoves instance\n"
        "history: Optional HistoryTable instance");
}

