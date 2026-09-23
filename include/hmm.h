#pragma once

#include <filesystem>
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

namespace fs = std::filesystem;

class HMM {
private:
    size_t n_states;
    bool ignore_warnings;
    
    Eigen::VectorXd start_priors;
    Eigen::VectorXd end_priors;
    Eigen::MatrixXd transp;
    Gaussian emissions;

    Eigen::VectorXd init_means_acc;
    Eigen::VectorXd init_std_dev_acc;
    Eigen::VectorXi init_obs_count_acc;
    
    Eigen::VectorXd start_prior_acc;
    Eigen::VectorXd end_prior_acc;
    
    Eigen::VectorXd gamma_acc;
    Eigen::MatrixXd xi_acc;
    
    Eigen::VectorXd obs_acc;
    Eigen::VectorXd obs_sq_acc;
    
    void forward(Eigen::MatrixXd& alpha, Eigen::VectorXd& c, const Eigen::MatrixXd& B);
    void backward(Eigen::MatrixXd& beta, Eigen::VectorXd& c, const Eigen::MatrixXd& B);
    void compute_gamma(Eigen::MatrixXd& gamma, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta);
    
    void compute_xi(Eigen::Tensor<double, 3>& xi, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& B);
    void update_accumulators(const Eigen::MatrixXd& gamma, const Eigen::Tensor<double, 3>& xi, const Eigen::VectorXd& obs);
    
public:
    HMM(size_t n_states, bool ignore_warnings) : n_states(n_states), ignore_warnings(ignore_warnings), emissions(n_states) {};

    // universal floor val to avoid 0 variance and div by zero
    inline static const double FLOOR_VAL = 1e-12;

    // set MIN FPS to 1. Here, FPS means frames per state rather
    // than frames per second. Although a little confusing, an abbreviation
    // keeps names compact
    inline static const size_t MIN_FRAMES_PER_STATE = 1;
    
    // load hmm definition from config
    void load_from_config(fs::path config_file);
    
    // initialize and public helpers
    // First, init_reset_accumulators should be called.
    // Then init_sequence should be called over each sequence.
    // Finally, init_accumulate should be called to gather
    // flat start init accumulators into the emissions.
    // void init_reset_accumulators();
    void init_accumulate(const Eigen::VectorXd& obs, const std::vector<std::string>& labels);
    void init_summarize();

    // baum welch and public helpers
    // First, baum_welch_reset_accumulators should be called.
    // Then baum_welch across all obs.
    // Then baum_welch_update_params. Calling the three in sequence
    // repeatedly forms multiple iterations of baum
    // welch.
    // void baum_welch_reset_accumulators();
    void baum_welch_accumulate(const Eigen::VectorXd& obs);
    void baum_welch_summarize();
    
    // getters
    const Eigen::VectorXd& get_start_priors() const { return start_priors; };
    const Eigen::VectorXd& get_end_priors() const { return end_priors; };
    const Eigen::MatrixXd& get_transp() const { return transp; };
    const Gaussian& get_emissions() const { return emissions; };
};

std::ostream& operator<<(std::ostream& os, const HMM& hmm);

