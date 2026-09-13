#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>

#include "utils.h"

namespace nb = nanobind;

// Define the nanobind module
NB_MODULE(utils, m) {
    nb::module_ cpp_logger_m = m.def_submodule("cpp_logger", "cpp logger module. encapsulates cpp logger functions.");

    cpp_logger_m.def("set_debug_logger", &utils::CppLogger::set_debug_cb);
    cpp_logger_m.def("set_info_logger", &utils::CppLogger::set_info_cb);
    cpp_logger_m.def("set_warning_logger", &utils::CppLogger::set_warning_cb);
    cpp_logger_m.def("set_error_logger", &utils::CppLogger::set_error_cb);
}

