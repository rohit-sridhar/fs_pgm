#include <sstream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "data_loader.h"
#include "hmm.h"
#include "utils.h"

namespace nb = nanobind;

NB_MODULE(pgm, m) {
    nb::module_ utils_m = m.def_submodule("utils", "utils module containing helper standalone functions and helper classes.");
    nb::module_ models_m = m.def_submodule("models", "models module containing pgm model classes.");
    nb::module_ cpp_logger_m = utils_m.def_submodule("cpp_logger", "cpp logger module. encapsulates cpp logger functions.");
    nb::module_ data_loader_m = utils_m.def_submodule("data_loader", "data loader module. encapsulates data loader classes and functions.");

    cpp_logger_m.def("set_debug_logger", &utils::CppLogger::set_debug_cb);
    cpp_logger_m.def("set_info_logger", &utils::CppLogger::set_info_cb);
    cpp_logger_m.def("set_warning_logger", &utils::CppLogger::set_warning_cb);
    cpp_logger_m.def("set_error_logger", &utils::CppLogger::set_error_cb);

    nb::class_<DataLoader>(data_loader_m, "DataLoader")
        .def(
            nb::init<const std::filesystem::path&>(),
            nb::arg("load_dir")
        )
        .def("load_data", &DataLoader::load_data);

    nb::class_<HMM>(models_m, "HMM")
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

