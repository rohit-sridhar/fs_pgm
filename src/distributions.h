#pragma once

// #include <nanobind/nanobind.h>
// #include <nanobind/eigen/dense.h>
#include <Eigen/Dense>

// namespace nb = nanobind;

class Gaussian {
public:
    Gaussian(size_t n_states);
    Eigen::MatrixXd pdf(const Eigen::VectorXd& obs);

    size_t n_states;
    Eigen::VectorXd means;
    Eigen::VectorXd std_devs;
};

std::ostream& operator<<(std::ostream& os, const Gaussian& g);

