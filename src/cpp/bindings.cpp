/*
 * bindings.cpp
 *
 * This file contains nanobind NB_MODULE command to package
 * various classes, static functions, and class methods into
 * one importable pgm package for python scripts to use.
 *
 * The tree defined in src/python defines the import structure
 * that is also reflected in the module declarations below.
 */

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
#include "trainer.h"

namespace nb = nanobind;

NB_MODULE(_pgm, m) {
    nb::module_ utils_m = m.def_submodule("utils", "utils module containing helper standalone functions and helper classes.");
    nb::module_ models_m = m.def_submodule("models", "models module containing pgm model classes.");
    nb::module_ trainer_m = m.def_submodule("trainer", "trainer module containing pgm hmm trainer class.");
    nb::module_ cpp_logger_m = utils_m.def_submodule("cpp_logger", "cpp logger module. encapsulates cpp logger functions.");
    nb::module_ data_loader_m = utils_m.def_submodule("data_loader", "data loader module. encapsulates data loader classes and functions.");

    cpp_logger_m.def("set_debug_logger", &utils::CppLogger::set_debug_cb);
    cpp_logger_m.def("set_info_logger", &utils::CppLogger::set_info_cb);
    cpp_logger_m.def("set_warning_logger", &utils::CppLogger::set_warning_cb);
    cpp_logger_m.def("set_error_logger", &utils::CppLogger::set_error_cb);

    // set data loader module expositions
    nb::class_<DataLoader>(data_loader_m, "DataLoader")
        .def(
            nb::init<const std::filesystem::path&>(),
            nb::arg("load_dir")
        )
        .def_ro_static(
            "DATAFILE_EXTENSION",
            &DataLoader::DATAFILE_EXTENSION
        )
        .def_ro_static(
            "LABFILE_EXTENSION",
            &DataLoader::LABFILE_EXTENSION
        )
        .def("__str__", [](const DataLoader& data_loader) {
            std::stringstream ss;
            ss << data_loader;
            return ss.str();
        })
        .def("__repr__", [](const DataLoader& data_loader) {
            std::stringstream ss;
            ss << data_loader;
            return ss.str();
        });

    // set hmm trainer module expositions
    nb::class_<HMMTrainer>(trainer_m, "HMMTrainer")
        .def(
            nb::init<HMM&, DataLoader&>(),
            nb::arg("hmm"),
            nb::arg("data_loader"),
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>()
        )
        .def(
            "initialize",
            &HMMTrainer::initialize
        )
        .def(
            "train",
            &HMMTrainer::train,
            nb::arg("iters") = 3
        )
        .def("__str__", [](const HMMTrainer& hmm_trainer) {
            std::stringstream ss;
            ss << hmm_trainer;
            return ss.str();
        })
        .def("__repr__", [](const HMMTrainer& hmm_trainer) {
            std::stringstream ss;
            ss << hmm_trainer;
            return ss.str();
        });
    
    // set hmm model module expositions
    nb::class_<HMM>(models_m, "HMM")
        .def(
            nb::init<size_t, bool>(),
            nb::arg("n_states"),
            nb::arg("ignore_warnings") = false
        )
        .def("load_from_config", &HMM::load_from_config)
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

