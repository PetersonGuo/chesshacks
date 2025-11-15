#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "functions.h"

namespace nb = nanobind;

NB_MODULE(c_helpers, m) {
    m.doc() = "ChessHacks C++ extension module";

    m.def("alpha_beta", []() {
    }, "A function for alpha-beta pruning");

    m.def("add", &add,
          nb::arg("a"), nb::arg("b"),
          "Add two integers");
}
