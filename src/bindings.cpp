#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "functions.h"

namespace nb = nanobind;

NB_MODULE(c_helpers, m) {
    m.doc() = "ChessHacks C++ extension module";

    m.def("alpha_beta", &alpha_beta, nb::arg("node"), nb::arg("depth"), nb::arg("alpha"), nb::arg("beta"), nb::arg("maximizingPlayer"),
          "A function for alpha-beta pruning");
}
