#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "distributions.h"

/*
 * HMM class
 *
 * The HMM class stores information about the markov model (num states,
 * start/end priors, transition probs and emission probs (as a Gaussian class member).
 *
 * Does forward/backward algorithm, computing alpha and beta and updates gamma and xi
 * to then update priors, transition probs and emissions in a vectorized manner.
 *
 * baum_welch is one iteration of forward backward and parameter updates.
 *
 * overrides the << stream operator so that printing hmm will print out the parameters
 * in multiple lines.
 */
class HMM {
private:
    size_t n_states;
    bool verbose;
    
    Eigen::VectorXd start_priors;
    Eigen::VectorXd end_priors;
    Eigen::MatrixXd transp;
    Gaussian emissions;
    
    void forward(Eigen::MatrixXd& alpha, Eigen::VectorXd& c, const Eigen::MatrixXd& B);
    void backward(Eigen::MatrixXd& beta, Eigen::VectorXd& c, const Eigen::MatrixXd& B);
    void compute_gamma(Eigen::MatrixXd& gamma, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta);
    
    void compute_xi(Eigen::Tensor<double, 3>& xi, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& B);
    void update_params(const Eigen::MatrixXd& gamma, const Eigen::Tensor<double, 3>& xi, const Eigen::VectorXd& obs);
    
public:
    HMM(size_t n_states, bool verbose) : n_states(n_states), verbose(verbose), emissions(n_states) {};
    
    void initialize_model();
    void baum_welch(const Eigen::VectorXd& obs);
    
    // getters
    const Eigen::VectorXd& get_start_priors() const { return start_priors; };
    const Eigen::VectorXd& get_end_priors() const { return end_priors; };
    const Eigen::MatrixXd& get_transp() const { return transp; };
    const Gaussian& get_emissions() const { return emissions; };
};

std::ostream& operator<<(std::ostream& os, const HMM& hmm);

