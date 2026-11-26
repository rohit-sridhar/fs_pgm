#define _USE_MATH_DEFINES
#include "distributions.h"
#include "utils.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/******************** Base Gaussian ********************/
// The base class constructor does nothing but set n_states
Gaussian::Gaussian(size_t n_states, size_t dim) : n_states_(n_states), dim_(dim) {
    means_ = Eigen::MatrixXd::Zero(n_states_, dim_);
}

// overload output stream operator to print params of gaussian
std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
    os << g.to_string();
    return os;
}


/******************** Single Dimensional Gaussian ********************/

/*
 * SingleDimensionalGaussian Constructor
 *
 * This constructor initializes zero vectors for means
 * and std devs based on the number of states in the HMM.
 */
SingleDimensionalGaussian::SingleDimensionalGaussian(size_t n_states) : Gaussian(n_states, 1) {
    set_std_devs(Eigen::VectorXd::Ones(n_states));
}

void SingleDimensionalGaussian::set_means(const Eigen::VectorXd& means) {
    if (means.size() != get_n_states())
        throw std::invalid_argument("error setting means: first dim must be n_states. last dim must be 1.");
    
    means_ = means;
}

void SingleDimensionalGaussian::set_std_devs(const Eigen::VectorXd& std_devs) {
    if (std_devs.size() != get_n_states())
        throw std::invalid_argument("error setting std devs: first dim must be n_states. last dim must be 1.");
    
    std_devs_ = std_devs;
}

/*
 * pdf (vectorized)
 * 
 * Vectorized computations of 1d normal distribution over grid of multiple
 * observations and each mean/std dev pair.
 *
 * Input X should be a column vector of length T.
 */
Eigen::MatrixXd SingleDimensionalGaussian::pdf(const Eigen::MatrixXd& X) const {
    if (X.cols() != 1) {
        std::string error_msg = "error. data passed to function must have 1 dims.";
        throw std::invalid_argument(error_msg);
    }

    auto X_vec = X.col(0);
    double scalar_numerator = 1.0 / std::sqrt(2.0 * M_PI);
    auto inv_std_devs = scalar_numerator / std_devs_.array();
    
    auto coefficients = inv_std_devs.replicate(1, X_vec.size());
    auto variances = std_devs_.array().square().replicate(1, X_vec.size());
    auto means_mat = means_.replicate(1, X_vec.size());
    
    auto diff = means_mat.rowwise() - X_vec.transpose();
    auto exponents_top = -(diff.array().square());
    auto exponents_bottom = 2.0 * variances;
    auto exponents = exponents_top.array() / exponents_bottom.array();
    
    Eigen::MatrixXd pdf = coefficients.array() * exponents.array().exp();
    return pdf;
}

std::string SingleDimensionalGaussian::to_string() const {
    std::ostringstream oss;
    oss << "means: \n\n"<< means_;
    oss << "\n\nstd devs (Col Major): \n\n" << std_devs_;
    return oss.str();
    
}

/******************** Multi Dimensional Gaussian ********************/
MultiDimensionalGaussian::MultiDimensionalGaussian(
    size_t n_states,
    size_t dim,
    bool diag
) : Gaussian(n_states, dim), diag_(diag), cov_stride_(n_states, n_states * dim) {
    means_ = Eigen::MatrixXd::Zero(n_states, dim);
    init_covs_();
}

/******************** Private Multi Dimensional Gaussian ********************/
void MultiDimensionalGaussian::init_covs_() {
    size_t n_states = get_n_states();
    size_t dim = get_dim();
    
    covs_ = Eigen::Tensor<double, 3>(n_states, dim, dim);
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (size_t i = 0; i < n_states; ++i) {
        TensorStrideMap cov(&covs_(i,0,0), dim, dim, cov_stride_);
        cov = 1.0 + ((Eigen::MatrixXd::Random(dim, dim).array() + 1.0) * 0.5);
        cov = cov.triangularView<Eigen::Lower>();
        cov = cov.transpose() * cov;
        log_debug() << "Cov " << i << ": " << cov << "\n";
        log_debug() << "Covs_ (chipped) " << i << ": " << covs_.chip(i,0) << "\n";
    }
}

/******************** Public Multi Dimensional Gaussian ********************/
/*
 * Computes PDF of MultiDimensional Gaussian
 *  X should be a matrix of size T x D
 *  return value is T x N
 */
Eigen::MatrixXd MultiDimensionalGaussian::pdf(const Eigen::MatrixXd& X) const {
    using namespace Eigen;
    size_t n_states = get_n_states();
    size_t dim = get_dim();
    
    if (X.cols() != dim) {
        std::string error_msg = std::format("error. data passed to function must have {} dims.", dim);
        throw std::invalid_argument(error_msg);
    }
    
    log_debug() << "X (Shape): (" << X.rows() << ", " << X.cols() << ")";
    MatrixXd probs(n_states, X.rows());
    log_debug() << "Probs (Shape): (" << probs.rows() << ", " << probs.cols() << ")";

    for(size_t i = 0; i < n_states; ++i) {
        // Get i-th state cov matrix and i-th state means
        auto means = means_.row(i);
        
        ConstTensorStrideMap cov(&covs_(i,0,0), dim, dim, cov_stride_);
        LLT<MatrixXd> llt(cov);
        if (llt.info() != Success) {
            std::string error_msg = std::format("Cov matrix for state {} is not invertible.", i);
            throw std::runtime_error(error_msg);
        }
        
        auto diffs = X - means.replicate(X.rows(), 1);
        auto diffs_cov = diffs * llt.solve(MatrixXd::Identity(dim, dim));
        auto exponent = (diffs_cov.array() * diffs.array()).rowwise().sum();

        double scalar = 1.0 / (std::pow(2.0 * M_PI, dim / 2.0) * std::sqrt(cov.determinant()));
        probs.row(i) = exponent.transpose().array() * scalar;
    }
    
    return probs;
}

/*
 *  set the covariance matrix
 */
void MultiDimensionalGaussian::set_covs(const Eigen::Tensor<double, 3, Eigen::ColMajor>& covs) {
    const auto& dims = covs.dimensions();
    size_t dim = get_dim();
    
    if (dims[0] != get_n_states() || dims[1] != dim || dims[2] != dim)
        throw std::invalid_argument("first dim must be n_states. last 2 dims must be size dim.");
    
    covs_ = covs;
}

// overload output stream operator to print params of gaussian
std::string MultiDimensionalGaussian::to_string() const {
    std::ostringstream oss;
    oss << "means: \n\n"<< means_;
    oss << "\n\nCov (Col Major): \n\n";
    
    for(size_t i = 0; i < get_n_states(); ++i) {
        oss << covs_.chip(i, 0) << "\n\n";
    }
    
    return oss.str();
}

