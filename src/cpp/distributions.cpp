#define _USE_MATH_DEFINES
#include "distributions.h"

#include <iomanip>
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

/*
 * Gaussian Constructor
 *
 * This constructor initializes zero vectors for means
 * and std devs based on the number of states in the HMM.
 */
Gaussian::Gaussian(size_t n_states, size_t dim, bool diag) : n_states(n_states), dim(dim), diag(diag) {
    means = Eigen::MatrixXd::Zero(n_states, dim);
    std_devs = Eigen::MatrixXd::Ones(n_states, dim);
}

Gaussian::Gaussian(size_t n_states) : n_states(n_states) {
    means = Eigen::VectorXd::Zero(n_states);
    std_devs = Eigen::VectorXd::Ones(n_states);
}

/*
 * pdf (vectorized)
 *
 * Vectorized computations of 1d normal distribution over grid of multiple
 * observations and each mean/std dev pair.
 */
Eigen::MatrixXd Gaussian::pdf(const Eigen::VectorXd& X) const {
    double scalar_numerator = 1.0 / std::sqrt(2.0 * M_PI);
    auto inv_std_devs = scalar_numerator / std_devs.array();
    
    auto coefficients = inv_std_devs.replicate(1, X.size());
    auto variances = std_devs.array().square().replicate(1, X.size());
    auto means_mat = means.replicate(1, X.size());
    
    auto diff = means_mat.rowwise() - X.transpose();
    auto exponents_top = -(diff.array().square());
    auto exponents_bottom = 2.0 * variances;
    auto exponents = exponents_top.array() / exponents_bottom.array();
    
    Eigen::MatrixXd pdf = coefficients.array() * exponents.array().exp();
    return pdf;
}

// overload output stream operator to print params of gaussian
std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
    os << "means: \n"<< g.means;
    os << "\nstd devs: \n" << g.std_devs;
    return os;
}

