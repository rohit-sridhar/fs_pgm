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

// The accumulators below work for initialization and baum welch.
// They accumulate (sums) of values in parallel and the HMM class
// aggregates (averages, or keeps the sum) them after the threads
// join. We keep these separate since each thread will work on its
// own copy.
struct HMMInitAccumulator {
    Eigen::VectorXd means;
    Eigen::VectorXd std_devs;
    Eigen::VectorXi obs_count;
    
    HMMInitAccumulator(size_t n_states) {
        means.setZero(n_states);
        std_devs.setZero(n_states);
        obs_count.setZero(n_states);
    }

    HMMInitAccumulator& operator+=(const HMMInitAccumulator& acc) {
        means += acc.means;
        std_devs += acc.std_devs;
        obs_count += acc.obs_count;
        return *this;
    }
};

struct HMMBaumWelchAccumulator {
    Eigen::VectorXd start_priors;
    Eigen::VectorXd end_priors;
    
    Eigen::VectorXd gamma;
    Eigen::MatrixXd xi;
    
    Eigen::VectorXd obs;
    Eigen::VectorXd obs_sq;

    HMMBaumWelchAccumulator(size_t n_states) {
        start_priors.setZero(n_states);
        end_priors.setZero(n_states);

        gamma.setZero(n_states);
        xi.setZero(n_states, n_states);

        obs.setZero(n_states);
        obs_sq.setZero(n_states);
    }

    HMMBaumWelchAccumulator& operator+=(const HMMBaumWelchAccumulator& acc) {
        start_priors += acc.start_priors;
        end_priors += acc.end_priors;

        gamma += acc.gamma;
        xi += acc.xi;

        obs += acc.obs;
        obs_sq += acc.obs_sq;
        return *this;
    }
};

class HMM {
private:
    size_t n_states;
    bool ignore_warnings;
    
    Eigen::VectorXd start_priors;
    Eigen::VectorXd end_priors;
    Eigen::MatrixXd transp;
    Gaussian emissions;

    void forward(Eigen::MatrixXd& alpha, Eigen::VectorXd& c, const Eigen::MatrixXd& B);
    void backward(Eigen::MatrixXd& beta, Eigen::VectorXd& c, const Eigen::MatrixXd& B);
    void compute_gamma(Eigen::MatrixXd& gamma, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta);
    
    void compute_xi(Eigen::Tensor<double, 3>& xi, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& B);
    void update_accumulators(const Eigen::MatrixXd& gamma, const Eigen::Tensor<double, 3>& xi, const Eigen::VectorXd& obs, HMMBaumWelchAccumulator& acc);
    
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
    void init_accumulate(const Eigen::VectorXd& obs, const std::vector<std::string>& labels, HMMInitAccumulator& acc);
    void init_summarize(HMMInitAccumulator& acc);

    // baum welch and public helpers
    // First, baum_welch_reset_accumulators should be called.
    // Then baum_welch across all obs.
    // Then baum_welch_update_params. Calling the three in sequence
    // repeatedly forms multiple iterations of baum
    // welch.
    // void baum_welch_reset_accumulators();
    void baum_welch_accumulate(const Eigen::VectorXd& obs, HMMBaumWelchAccumulator& acc);
    void baum_welch_summarize(HMMBaumWelchAccumulator& acc);
    
    // getters
    size_t get_n_states() const { return n_states; };
    const Eigen::VectorXd& get_start_priors() const { return start_priors; };
    const Eigen::VectorXd& get_end_priors() const { return end_priors; };
    const Eigen::MatrixXd& get_transp() const { return transp; };
    const Gaussian& get_emissions() const { return emissions; };
};

std::ostream& operator<<(std::ostream& os, const HMM& hmm);

