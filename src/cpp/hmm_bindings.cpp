#include <sstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "hmm.h"

namespace nb = nanobind;

// Define the nanobind module
NB_MODULE(hmm, m) {
    nb::class_<HMM>(m, "HMM")
        .def(
            nb::init<size_t, bool>(),
            nb::arg("n_states"),
            nb::arg("verbose") = false
        )
        .def("initialize_model", &HMM::initialize_model)
        .def("baum_welch", &HMM::baum_welch)
        .def("__str__", [](const HMM& hmm) {
            std::stringstream ss;
            ss << hmm;
            return ss.str();
        })
        .def("__repr__", [](const HMM& hmm) {
            std::stringstream ss;
            ss << hmm;
            return ss.str();
        });
}

