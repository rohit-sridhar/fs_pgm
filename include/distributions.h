#pragma once

#include <Eigen/Dense>

class Gaussian {
public:
    Gaussian(size_t n_states);
    Eigen::MatrixXd pdf(const Eigen::VectorXd& obs);

    size_t n_states;
    Eigen::VectorXd means;
    Eigen::VectorXd std_devs;
};

std::ostream& operator<<(std::ostream& os, const Gaussian& g);

