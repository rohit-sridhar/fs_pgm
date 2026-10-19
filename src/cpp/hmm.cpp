#include "hmm.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <array>
#include <stdexcept>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "distributions.h"
#include "data_loader.h"
#include "utils.h"

/**************************************** CONSTRUCTOR ****************************************/
/*
 * HMM Constructor. First, Checks validity of n_states (can't be 0 and must be a multiple of
 * len(labels)). Then sets states per label to the positive int states_per_label. Finally,
 * sets the reverse map (label -> index) from the forward map, label_vec (index -> label).
 */
HMM::HMM(
    size_t n_states,
    std::vector<std::string>& labels,
    bool ignore_warnings
) : n_states(n_states),
    label_vec(labels),
    ignore_warnings(ignore_warnings),
    emissions(n_states) {
    
    if (n_states == 0) {
        log_error() << "n_states must be > 0";
        throw std::invalid_argument("n_states must be > 0");
    }
    
    if (n_states % label_vec.size() != 0) {
        log_error() << "The number of labels must divide n_states with no remainder";
        throw std::invalid_argument("The number of labels must divide n_states with no remainder");
    }
    
    states_per_label = n_states / label_vec.size();
    for (size_t i = 0; i < label_vec.size(); ++i) {
        label_map[label_vec[i]] = i;
    }
}

/**************************************** PRIVATE FXN ****************************************/

/*
 * Run forward algorithm
 *
 * We fill the first col of alpha using the emissions matrix
 * and start priors. Later cols use the transition probability
 * from previous possible states, the emissions matrix and 
 * the inductive probability computed until that point (as a prior).
 *
 * We scale alpha at each timestep (normalizing by the col sum)
 * and store it in Eigen::VectorXd c. It is used by the backward
 * algorithm as well. When gamma and xi are computed, the scalars
 * cancel so that parameter updates are unaffected.
 */
void HMM::forward(Eigen::MatrixXd& alpha, Eigen::VectorXd& c, const Eigen::MatrixXd& B) const {
    alpha.col(0) = B.col(0).array() * start_priors.array();
    
    // scale alpha
    c(0) = 1.0 / alpha.col(0).sum();
    alpha.col(0) *= c(0);

    for (size_t t = 1; t < alpha.cols(); ++t) {
        auto prev = transp.transpose() * alpha.col(t - 1);
        alpha.col(t) = B.col(t).array() * prev.array();
        
        // scale alpha
        c(t) = 1.0 / alpha.col(t).sum();
        alpha.col(t) *= c(t);
    }
}

/*
 * Run backward algorithm
 *
 * We fill the final col of beta using the alpha scalar value c(T)
 * and the end priors array.
 *
 * We iterate backwards over timesteps computing the new col of beta
 * using transition probabilities of transitioning to the next state,
 * emissions for the next state and the inductive prior from the next state.
 * We then scale that column of beta using c(t).
 */
void HMM::backward(Eigen::MatrixXd& beta, Eigen::VectorXd& c, const Eigen::MatrixXd& B) const {
    size_t T = beta.cols() - 1;
    beta.col(T) = c(T) * end_priors.array();

    for (size_t t = T - 1; t != SIZE_MAX; --t) {
        auto next = beta.col(t + 1).array() * B.col(t + 1).array();
        
        beta.col(t) = transp * next.matrix();
        beta.col(t) *= c(t);
    }
}

/*
 * Compute gamma
 *
 * Gamma is the probability of passing through a given state
 * for the sequence being examined. It combined forwards and backwards
 * probabilities to compute the probability of being in a given state
 * at a timestep t for that sequence (considering all state sequences passing
 * throuh that state).
 */
void HMM::compute_gamma(Eigen::MatrixXd& gamma, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta) const {
    gamma = alpha.array() * beta.array();
    gamma.array().rowwise() /= gamma.colwise().sum().array();
}

/*
 * Compute xi over every timestep in the observation sequence.
 */
void HMM::compute_xi(Eigen::Tensor<double, 3>& xi, const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta, const Eigen::MatrixXd& B) const {
    using namespace Eigen;
    
    size_t T = xi.dimension(0);
    size_t num_lowsums = 0;
    
    for(size_t t = 0; t < T; ++t) {
        auto trans_to = beta.col(t + 1).array() * B.col(t + 1).array();
        auto trans_from = transp.array().colwise() * alpha.col(t).array();
        
        // TensorMap<Tensor<double, 2>> xi_t(&xi(0, 0, t), n_states, n_states);
        MatrixXd xi_t = trans_from.array().rowwise() * trans_to.array().transpose();
        double total_sum = xi_t.sum();
        
        if (total_sum > HMM::FLOOR_VAL)
            xi_t /= total_sum;
        else {
            // Fallback strategy to protect the rest of the matrix math
            xi_t.setConstant(1.0 / (n_states * n_states));
            num_lowsums++;
        }
        
        xi.chip(t, 0) = TensorMap<const Tensor<const double, 2>>(
            xi_t.data(),
            xi_t.rows(),
            xi_t.cols()
        );
    }
    
    // if (num_lowsums > 0 && !ignore_warnings) {
    //     log_warning() << "Baum-Welch Underflow/Impossible Path at " << num_lowsums << " timesteps. "
    //               << "The observation sequence cannot be generated by the current model parameters. "
    //               << "Assigning uniform fallback to avoid NaNs.\n";
    // }
}

/*
 * update accumulators
 */
void HMM::update_accumulators(
    const Eigen::MatrixXd& gamma,
    const Eigen::Tensor<double, 3>& xi,
    const Eigen::VectorXd& obs,
    HMMBaumWelchAccumulator& acc
) const {
    using namespace Eigen;

    acc.start_priors += gamma.col(0);
    acc.end_priors += gamma.col(gamma.cols() - 1);
    
    acc.gamma += gamma.rowwise().sum();
    std::array<int, 1> time_dim = {0};
    Tensor<double, 2> xi_sum_t = xi.sum(time_dim);
    Map<MatrixXd> xi_sum(xi_sum_t.data(), xi.dimension(1), xi.dimension(2));
    acc.xi += xi_sum;
    
    acc.obs.noalias() += (gamma * obs);
    acc.obs_sq.noalias() += (gamma * obs.array().square().matrix());
}

/**************************************** PUBLIC FXN ****************************************/

/*
 * Loads hmm from config file
 * For now, config file should contain
 *  -- start priors
 *  -- end priors
 *  -- transp matrix
 */
void HMM::load_from_config(fs::path config_file) {
    using namespace Eigen;

    log_info() << "Loading config file: " << config_file.filename();
    std::ifstream file(config_file); 
    
    if (!file.is_open()) {
        log_error() << "Error: Could not open file at " << config_file;
        log_error() << "Could not load HMM config.";
        return;
    }
    
    std::string line;
    std::vector<double> flat_data;
    
    size_t num_rows = 0;
    size_t num_cols = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) {
            log_error() << "Error: Found empty line in config file " << config_file;
            log_error() << "Could not load HMM config.";
            return;
        }
        
        std::stringstream ss(line);
        double value;
        size_t current_row_cols = 0;
        
        while (ss >> value) {
            flat_data.push_back(value);
            current_row_cols++;
        }
        
        if (current_row_cols != n_states) {
            log_error() << "Error. Config must have " << n_states << " cols.";
            log_error() << "Could not load " << config_file;
            return;
        }
        num_rows++;
    }

    if (num_rows - 2 != n_states) {
        log_error() << "Error. Config must have " << n_states + 2 << " rows.";
        log_error() << "Could not load " << config_file;
        return;
    }
    
    // Map first row into start priors
    const double* start = flat_data.data();
    start_priors = Map<const VectorXd>(start, n_states);
    
    // Map next row into end priors
    start += n_states;
    end_priors = Map<const VectorXd>(start, n_states);
    
    // Map and copy the transp table
    start += n_states;
    using mmatrix = Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>;
    transp = mmatrix(start, n_states, n_states);
    
    log_info() << "Successfully loaded start priors, end priors and transp from: " << config_file.filename();
}

/*
 * Initialize the Hidden Markov Model. For now it just initializes
 * emissions. The user is expected to initialize transp and start/end
 * priors.
 *
 * The user should call init_reset_accumulators before calling
 * this function.
 */
void HMM::init_accumulate(
    const Eigen::VectorXd& obs,
    const std::vector<std::string>& labels,
    HMMInitAccumulator& acc
) const {
    // TODO add this to all functions using Eigen
    using namespace Eigen;
    
    // fpl is frames per label and fps is frames per state.
    // 1 label may correspond to a state or multiple.
    size_t fpl = obs.size() / labels.size();
    size_t fps = obs.size() / n_states;
    
    if (fps <= HMM::MIN_FRAMES_PER_LABEL)
        std::invalid_argument("There are too few observations per label. Skipping this sequence.");
    if (fps <= HMM::MIN_FRAMES_PER_STATE)
        std::invalid_argument("There are too few observations per state. Skipping this sequence.");
    
    for (size_t i = 0; i < labels.size(); ++i) {
        size_t start_obs = i * fpl;
        size_t end_obs = start_obs + fpl - 1;
        if (i == labels.size() - 1)
            end_obs = obs.size() - 1;
        
        size_t start_state = label_map.at(labels[i]) * states_per_label;
        size_t end_state = start_state + states_per_label - 1;
        
        Refd acc_means = acc.means(seq(start_state, end_state));
        Refd acc_std_devs = acc.std_devs(seq(start_state, end_state));
        Refi acc_obs_count = acc.obs_count(seq(start_state, end_state));
        ConstRefd obs_seg = obs(seq(start_obs, end_obs));
        
        acc_means.array() += obs_seg.sum();
        auto acc_means_mat = acc_means.replicate(1, obs_seg.size());
        auto obs_seg_mat = obs_seg.transpose().replicate(acc_means.size(), 1);
        auto mean_diffs = acc_means_mat - obs_seg_mat;

        acc_std_devs.array() += mean_diffs.array().square().rowwise().sum();
        // we don't add by fpl here because the last segment may be longer
        acc_obs_count.array() += (end_obs - start_obs + 1);
    }
    
    // iterate until last state 
    // for (size_t i = 0; i < n_states; ++i) {
    //     size_t start = i * seglen;
    //     // since seglen is originally from floor division
    //     // we grab the rest of the array for the last iter.
    //     if (i == n_states - 1)
    //         seglen = obs.size() - start;
    //     
    //     acc.means(i) += obs.segment(start, seglen).sum();
    //     VectorXd mean_diffs = obs.segment(start, seglen).array() - acc.means(i);
    //     acc.std_devs(i) += mean_diffs.array().square().sum();
    //     acc.obs_count(i) += seglen;
    // }
}

/*
 * Updates emissions mean and std dev params.
 */
void HMM::init_summarize(HMMInitAccumulator& acc) {
    auto& acc_obs_count_dbl = acc.obs_count.cast<double>();
    emissions.means = acc.means.array() / acc_obs_count_dbl.array();
    emissions.std_devs = (acc.std_devs.array() / acc_obs_count_dbl.array()).array().sqrt();
}

/*
 * Single iteration of baum welch. Takes an
 * observation and accumulates gamma, xi and obs/obs^2.
 */
void HMM::baum_welch_accumulate(
    const Eigen::VectorXd& obs,
    HMMBaumWelchAccumulator& acc
) const {
    using namespace Eigen;
    size_t T = obs.size();
    
    MatrixXd alpha = MatrixXd::Zero(n_states, T);
    MatrixXd beta = MatrixXd::Zero(n_states, T);
    
    MatrixXd B = emissions.pdf(obs);
    
    VectorXd c = VectorXd::Zero(T);
    forward(alpha, c, B);
    backward(beta, c, B);
    
    MatrixXd gamma = MatrixXd::Zero(n_states, T);
    compute_gamma(gamma, alpha, beta);
    
    Tensor<double, 3> xi(T - 1, n_states, n_states);
    compute_xi(xi, alpha, beta, B);
    
    update_accumulators(gamma, xi, obs, acc);
}

/*
 * Updates params using accumulators. baum welch
 * automatically updates the accumulators by aggregating
 * data. This step fills in params after an entire batch
 * has been accumulated.
 */
void HMM::baum_welch_summarize(
    HMMBaumWelchAccumulator& acc
) {
    using namespace Eigen;

    start_priors = acc.start_priors / acc.start_priors.sum();
    end_priors = acc.end_priors / acc.end_priors.sum();

    VectorXd transp_norm = (acc.gamma.array() - acc.end_priors.array()).cwiseMax(HMM::FLOOR_VAL);
    transp = acc.xi.array().colwise() / transp_norm.array();
    
    emissions.means = acc.obs.array() / acc.gamma.array();
    VectorXd variances = (acc.obs_sq.array() / acc.gamma.array()) - emissions.means.array().square();
    log_debug() << "variances \n" << variances.cwiseMax(HMM::FLOOR_VAL);
    emissions.std_devs = variances.cwiseMax(HMM::FLOOR_VAL).array().sqrt();
}

// overload ostream for clean print
std::ostream& operator<<(std::ostream& os, const HMM& hmm) {
    os << "Start prior probabilities: \n";
    os << hmm.get_start_priors() << "\n\n";

    os << "End prior probabilities: \n";
    os << hmm.get_end_priors() << "\n\n";

    os << "Transition probabilities \n";
    os << hmm.get_transp() << "\n\n";
    
    os << "Emission probabilities: \n";
    os << hmm.get_emissions() << "\n\n";

    os << "Label set: \n";
    os << hmm.get_labels() << "\n\n";
    return os;
}

