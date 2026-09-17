#pragma once

#include <Eigen/Dense>

/*
 * Gaussian class
 *
 * A class for modeling gaussians. Stores means and std devs
 * as Eigen::Vectors and does Vectorized pdf computations for
 * a vector of observations across all clusters.
 *
 * Overrides stream << operator to print the mean and std dev
 * Eigen::Vectors out.
 */
class Gaussian {
public:
    Gaussian(size_t n_states);
    Eigen::MatrixXd pdf(const Eigen::VectorXd& obs);

    size_t n_states;
    Eigen::VectorXd means;
    Eigen::VectorXd std_devs;
};

std::ostream& operator<<(std::ostream& os, const Gaussian& g);

