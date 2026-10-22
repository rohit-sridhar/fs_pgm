#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

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
    Eigen::MatrixXd pdf(const Eigen::VectorXd& X) const;
    
    size_t n_states;
    Eigen::VectorXd means;
    Eigen::VectorXd std_devs;
};

/*
 * MultdimensionalGaussian class def
 */
class MultidimensionalGaussian {
public:
    using DynamicStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using TensorStrideMap = Eigen::Map<Eigen::MatrixXd, 0, DynamicStride>;
    using ConstTensorStrideMap = Eigen::Map<const Eigen::MatrixXd, 0, DynamicStride>;
    const Eigen::MatrixXd& means = priv_means;
    const Eigen::Tensor<double, 3>& covs = priv_covs;
    
    MultidimensionalGaussian(size_t n_states, size_t dim, bool diag);
    Eigen::MatrixXd pdf(const Eigen::MatrixXd& X) const;

    void set_covs(const Eigen::Tensor<double, 3, Eigen::ColMajor>& covs);
private:
    size_t n_states;
    size_t dim;
    bool diag;
    
    Eigen::MatrixXd priv_means;
    Eigen::Tensor<double, 3> priv_covs;
};

std::ostream& operator<<(std::ostream& os, const Gaussian& g);
std::ostream& operator<<(std::ostream& os, const MultidimensionalGaussian& mg);

