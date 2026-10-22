#define _USE_MATH_DEFINES
#include "distributions.h"
#include "utils.h"

#include <iomanip>
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/******************** Multi Dimensional Gaussian ********************/
MultidimensionalGaussian::MultidimensionalGaussian(
    size_t n_states,
    size_t dim,
    bool diag
) : n_states(n_states), dim(dim), diag(diag) {
    priv_means = Eigen::MatrixXd::Zero(n_states, dim);
    
    priv_covs = Eigen::Tensor<double, 3>(n_states, dim, dim);
    for (size_t i = 0; i < n_states; ++i) {
        DynamicStride stride(n_states * dim, n_states);
        TensorStrideMap cov(&priv_covs(i,0,0), dim, dim, stride);
        if (n_states == 0)
            cov <<  0.0, 1.0, 0.0,
                    1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0;
        else if (n_states == 1)
            cov <<  1, 0, 0,
                    0, 0, 1,
                    0, 1, 0;
        else
            cov <<  0, 0, 1,
                    1, 0, 0,
                    0, 1, 0;
    }
}

/*
 * Computes PDF of Multidimensional Gaussian
 *  X should be a matrix of size T x D
 *  return value is T x N
 */
Eigen::MatrixXd MultidimensionalGaussian::pdf(const Eigen::MatrixXd& X) const {
    using namespace Eigen;
    
    MatrixXd probs(X.rows(), n_states);
    for(size_t i = 0; i < n_states; ++i) {
        // Get i-th state cov matrix and i-th state means
        auto means = priv_means.row(i);
        
        DynamicStride stride(n_states * dim, n_states);
        ConstTensorStrideMap cov(&priv_covs(i,0,0), dim, dim, stride);
        
        auto diffs = X - means.replicate(X.rows(), 1);
        auto diffs_cov = diffs * cov;
        auto exponent = (diffs_cov.array() * diffs.array()).rowwise().sum();
        
        double scalar = 1.0 / (std::pow(2.0 * M_PI, dim / 2.0) * std::sqrt(cov.determinant()));
        probs.col(i) = exponent.array() * scalar;
    }
    
    return probs;
}

/*
 *  set the covariance matrix
 */
void MultidimensionalGaussian::set_covs(const Eigen::Tensor<double, 3, Eigen::ColMajor>& covs) {
    const auto& dims = covs.dimensions();
    
    if (dims[0] != n_states || dims[1] != dim || dims[2] != dim)
        throw std::invalid_argument("first dim must be n_states. last 2 dims must be size dim.");
    
    priv_covs = covs;
}

// overload output stream operator to print params of gaussian
std::ostream& operator<<(std::ostream& os, const MultidimensionalGaussian& mg) {
    os << "means: \n"<< mg.means;
    os << "\nstd devs: \n" << mg.covs;
    return os;
}

/******************** Single Dimensional Gaussian ********************/

/*
 * Gaussian Constructor
 *
 * This constructor initializes zero vectors for means
 * and std devs based on the number of states in the HMM.
 */
Gaussian::Gaussian(size_t n_states) : n_states(n_states) {
    means = Eigen::VectorXd::Zero(n_states);
    std_devs = Eigen::VectorXd::Ones(n_states);
}

/*
 * pdf (vectorized)
 * 
 * Vectorized computations of 1d normal distribution over grid of multiple
 * observations and each mean/std dev pair.
 *
 * Input X should be a column vector of length T.
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
    os << "\nstd devs (Col Major): \n" << g.std_devs;
    return os;
}

