#define _USE_MATH_DEFINES

#include <iomanip>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "distributions.h"

//private
// None for now

// public:
// Gaussian::Gaussian(double mu, double std_dev) : mu(mu), std_dev(std_dev) {}
Gaussian::Gaussian(size_t n_states) : n_states(n_states) {
    means = Eigen::VectorXd::Zero(n_states);
    std_devs = Eigen::VectorXd::Ones(n_states);
}

// overload output stream operator to print params of gaussian
std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
    os << "means: \n"<< g.means;
    os << "\nstd devs: \n" << g.std_devs;
    return os;
}

Eigen::MatrixXd Gaussian::pdf(const Eigen::VectorXd& X) {
    double scalar_numerator = 1.0 / std::sqrt(2.0 * M_PI);
    Eigen::VectorXd inv_std_devs = scalar_numerator / std_devs.array();

    Eigen::MatrixXd coefficients = inv_std_devs.replicate(1, X.size());
    Eigen::MatrixXd variances = std_devs.array().square().replicate(1, X.size());
    Eigen::MatrixXd means_mat = means.replicate(1, X.size());

    Eigen::MatrixXd diff = means_mat.rowwise() - X.transpose();
    Eigen::MatrixXd exponents_top = -(diff.array().square());
    Eigen::MatrixXd exponents_bottom = 2.0 * variances;
    Eigen::MatrixXd exponents = exponents_top.array() / exponents_bottom.array();

    Eigen::MatrixXd pdf = coefficients.array() * exponents.array().exp().array();
    return pdf;
}

