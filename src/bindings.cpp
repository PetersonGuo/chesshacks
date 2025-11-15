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

  // Original alpha_beta (creates its own transposition table)
  m.def("alpha_beta", &alpha_beta, nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), nb::arg("maximizingPlayer"),
        nb::arg("evaluate"),
        "Alpha-beta pruning search on chess position given as FEN string.\n"
        "evaluate: Python callback function that takes a FEN string and "
        "returns an int score");

  // Alpha-beta with persistent transposition table
  m.def("alpha_beta_with_tt", &alpha_beta_with_tt, nb::arg("fen"),
        nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("evaluate"), nb::arg("tt"),
        "Alpha-beta pruning with reusable transposition table.\n"
        "tt: TranspositionTable instance to cache positions across multiple "
        "searches");
  
  // Parallel alpha-beta (creates its own transposition table)
  m.def("alpha_beta_parallel", &alpha_beta_parallel, nb::arg("fen"),
        nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("evaluate"),
        nb::arg("num_threads") = 0,
        "Parallel alpha-beta pruning search (evaluates root moves in parallel).\n"
        "num_threads: Number of threads to use (0 = auto-detect)");
  
  // Parallel alpha-beta with persistent transposition table
  m.def("alpha_beta_parallel_with_tt", &alpha_beta_parallel_with_tt, 
        nb::arg("fen"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"),
        nb::arg("maximizingPlayer"), nb::arg("evaluate"), nb::arg("tt"),
        nb::arg("num_threads") = 0,
        "Parallel alpha-beta with shared transposition table.\n"
        "tt: Thread-safe TranspositionTable\n"
        "num_threads: Number of threads to use (0 = auto-detect)");
}

