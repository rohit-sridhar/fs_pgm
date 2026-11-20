#pragma once

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

class Gaussian {
public:
    virtual ~Gaussian() = default;
    Gaussian(size_t n_states, size_t dim);

    virtual Eigen::MatrixXd pdf(const Eigen::VectorXd& X) const = 0;
     
    // shared getters
    const Eigen::MatrixXd& get_means() const { return means_; };
    size_t get_n_states() const { return n_states_; };
    size_t get_dim() const { return dim_; };

    // admin functions
    virtual std::string to_string() const = 0;

protected:
    Eigen::MatrixXd means_;
    
private:
    const size_t n_states_;
    const size_t dim_;

};

/*
 * SingleDimensionalGaussian class
 *
 * A class for modeling gaussians. Stores means and std devs
 * as Eigen::Vectors and does Vectorized pdf computations for
 * a vector of observations across all clusters.
 *
 * Overrides stream << operator to print the mean and std dev
 * Eigen::Vectors out.
 */
class SingleDimensionalGaussian : public Gaussian {
public:
    SingleDimensionalGaussian(size_t n_states);
    Eigen::MatrixXd pdf(const Eigen::VectorXd& X) const override;
    
    // additional setters (define in cpp file since they do validation)
    void set_means(const Eigen::VectorXd& means);
    void set_std_devs(const Eigen::VectorXd& std_devs);

    // additional getters
    const Eigen::VectorXd& get_std_devs() const { return std_devs_; };

    // admin functions
    std::string to_string() const override;

protected:
    Eigen::VectorXd std_devs_;
};

/*
 * MultiDimensionalGaussian class def
 */
class MultiDimensionalGaussian {
public:
    MultiDimensionalGaussian(size_t n_states, size_t dim, bool diag);
    
    using DynamicStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using TensorStrideMap = Eigen::Map<Eigen::MatrixXd, 0, DynamicStride>;
    using ConstTensorStrideMap = Eigen::Map<const Eigen::MatrixXd, 0, DynamicStride>;
    
    const Eigen::MatrixXd& means = priv_means;
    const Eigen::Tensor<double, 3>& covs = priv_covs;
    
    Eigen::MatrixXd pdf(const Eigen::MatrixXd& X) const;
    void set_covs(const Eigen::Tensor<double, 3, Eigen::ColMajor>& covs);
    
    size_t get_n_states() const { return n_states; };
    size_t get_dim() const { return dim; };
private:
    size_t n_states;
    size_t dim;
    bool diag;
    
    DynamicStride cov_stride;
    Eigen::MatrixXd priv_means;
    Eigen::Tensor<double, 3> priv_covs;

    // returns standard col major stride for mv cov matrix
    // DynamicStride get_cov_stride() const { return DynamicStride(n_states * dim, n_states); };
};

std::ostream& operator<<(std::ostream& os, const Gaussian& g);
std::ostream& operator<<(std::ostream& os, const MultiDimensionalGaussian& mg);

