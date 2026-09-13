#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "data_loader.h"

namespace nb = nanobind;

// Define the nanobind module
NB_MODULE(data_loader, m) {
    nb::class_<DataLoader>(m, "DataLoader")
        .def(
            nb::init<const std::filesystem::path&>(),
            nb::arg("load_dir")
        )
        .def("load_data", &DataLoader::load_data);
}

