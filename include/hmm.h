#pragma once

#include <vector>
#include <iostream>
#include <string>
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

struct HMMSVInitAccumulator;
struct HMMSVBaumWelchAccumulator;

// The accumulators below work for initialization and baum welch.
// They accumulate (sums) of values in parallel and the HMM class
// aggregates (averages, or keeps the sum) them after the threads
// join. We keep these separate since each thread will work on its
// own copy.
struct HMMInitAccumulator {
    Eigen::VectorXi obs_count;
    size_t n_states;
    
    HMMInitAccumulator(size_t n_states) : n_states(n_states) {
        clear();
    }

    virtual void clear() {
        obs_count.setZero(n_states);
    }

    virtual HMMInitAccumulator& operator+=(const HMMInitAccumulator& acc) {
        obs_count += acc.obs_count;
        return *this;
    }
};

struct HMMBaumWelchAccumulator {
    Eigen::VectorXd start_priors;
    Eigen::VectorXd end_priors;
    
    Eigen::VectorXd gamma;
    Eigen::MatrixXd xi;
    
    size_t n_states;
    
    HMMBaumWelchAccumulator(size_t n_states) : n_states(n_states) {
        clear();
    }
    
    void clear() {
        start_priors.setZero(n_states);
        end_priors.setZero(n_states);
        
        gamma.setZero(n_states);
        xi.setZero(n_states, n_states);
    }

    HMMBaumWelchAccumulator& operator+=(const HMMBaumWelchAccumulator& acc) {
        start_priors += acc.start_priors;
        end_priors += acc.end_priors;

        gamma += acc.gamma;
        xi += acc.xi;
        
        return *this;
    }
};

class HMM {
public:
    HMM(size_t n_states, std::vector<std::string>& labels);

    // EigenRef declarations for zero copy references
    using Refd = Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using ConstRefd = Eigen::Ref<const Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using Refi = Eigen::Ref<Eigen::MatrixXi, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    
    // universal floor val to avoid 0 variance and div by zero
    inline static const double FLOOR_VAL = 1e-12;
    
    // set MIN FPS and MIN FPL to 1. Here, FPS means frames per state rather
    // than frames per second. Although a little confusing, an abbreviation
    // keeps names compact. FPL is frames per Label.
    inline static const size_t MIN_FRAMES_PER_LABEL = 1;
    inline static const size_t MIN_FRAMES_PER_STATE = 1;
    
    // load hmm definition from config
    void load_from_config(fs::path config_file);
    
    // initialize and public helpers
    // First, init_reset_accumulators should be called.
    // Then init_sequence should be called over each sequence.
    // Finally, init_accumulate should be called to gather
    // flat start init accumulators into the emissions.
    // void init_reset_accumulators();
    void init_accumulate(const Eigen::VectorXd& obs, const std::vector<std::string>& labels, HMMSVInitAccumulator& acc) const;
    void init_summarize(HMMSVInitAccumulator& acc);
    // virtual void init_accumulate(const Eigen::VectorXd& obs, const std::vector<std::string>& labels, HMMInitAccumulator& acc) const = 0;
    // virtual void init_summarize(HMMInitAccumulator& acc) = 0;
    
    // baum welch and public helpers
    // First, baum_welch_reset_accumulators should be called.
    // Then baum_welch across all obs.
    // Then baum_welch_update_params. Calling the three in sequence
    // repeatedly forms multiple iterations of baum welch.
    // void baum_welch_reset_accumulators();
    void baum_welch_accumulate(const Eigen::VectorXd& obs, HMMSVBaumWelchAccumulator& acc) const;
    void baum_welch_summarize(HMMSVBaumWelchAccumulator& acc);
    // virtual void baum_welch_summarize(HMMBaumWelchAccumulator& acc) = 0;
    
    // getters
    size_t get_n_states() const { return n_states_; };
    
    const Eigen::VectorXd& get_start_priors() const { return start_priors_; };
    const Eigen::VectorXd& get_end_priors() const { return end_priors_; };
    
    const Eigen::MatrixXd& get_transp() const { return transp_; };
    const Gaussian& get_emissions() const { return emissions_; };
    const std::vector<std::string>& get_labels() const { return label_vec_; };

    HMMSVInitAccumulator create_init_accumulator() const;
    HMMSVBaumWelchAccumulator create_baum_welch_accumulator() const;

private:
    size_t n_states_;
    size_t states_per_label_;
    
    // label_vec_ is a vector of labels
    // label_map_ maps labels in label_vec_ to indices in label_vec_
    std::vector<std::string> label_vec_;
    std::unordered_map<std::string, size_t> label_map_;
    
    Eigen::VectorXd start_priors_;
    Eigen::VectorXd end_priors_;
    Eigen::MatrixXd transp_;
    SingleDimensionalGaussian emissions_;
    
    void forward_(Eigen::MatrixXd& alpha, Eigen::VectorXd& c, const Eigen::MatrixXd& B) const;
    void backward_(Eigen::MatrixXd& beta, Eigen::VectorXd& c, const Eigen::MatrixXd& B) const;
    
    void compute_gamma_(Eigen::MatrixXd& gamma, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta) const;
    void compute_xi_(Eigen::Tensor<double, 3>& xi, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& B) const;
    
    void update_accumulators_(const Eigen::MatrixXd& gamma, const Eigen::Tensor<double, 3>& xi, const Eigen::VectorXd& obs, HMMSVBaumWelchAccumulator& acc) const;    
};

// class HMMSingleGaussian : public HMM {
// //     public:
// //         void init_accumulate(const Eigen::VectorXd& obs, const std::vector<std::string>& labels, HMMInitAccumulator& acc) const override;
// //         void init_summarize(HMMInitAccumulator& acc) override;
// //         virtual void baum_welch_summarize(HMMBaumWelchAccumulator& acc) override;
// };

std::ostream& operator<<(std::ostream& os, const HMM& hmm);

