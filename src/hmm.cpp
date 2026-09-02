#include <nanobind/nanobind.h>
#include <iostream>

namespace nb = nanobind;

// 1. Define your standard C++ Class
class HMM {
public:
    // No-argument constructor
    HMM() {}

    // Test function
    void print_test() {
        std::cout << "Hello World from the C++ Backend!" << std::endl;
    }
};

// 2. Define the nanobind module
NB_MODULE(hmm, m) {
    nb::class_<HMM>(m, "HMM")
        .def(nb::init<>()) // Binds the no-argument constructor
        .def("print_test", &HMM::print_test); // Binds the method
}

