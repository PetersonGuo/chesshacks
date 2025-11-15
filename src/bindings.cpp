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

  m.def("alpha_beta", &alpha_beta, nb::arg("fen"), nb::arg("depth"),
        nb::arg("alpha"), nb::arg("beta"), nb::arg("maximizingPlayer"),
        nb::arg("evaluate"),
        "Alpha-beta pruning search on chess position given as FEN string.\n"
        "evaluate: Python callback function that takes a FEN string and "
        "returns an int score");
}
