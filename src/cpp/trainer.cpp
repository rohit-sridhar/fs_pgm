#include "trainer.h"

#include <Eigen/Dense>
#include <omp.h>
#include <thread>

#include "hmm.h"
#include "hmm_sv_gaussian.h"
#include "data_loader.h"
#include "utils.h"

HMMTrainer::HMMTrainer(
    HMM& hmm,
    DataLoader& data_loader,
    size_t batch_size
) : hmm_(hmm), data_loader_(data_loader), max_batch_files_(batch_size) {
    // Query OS-allocated core count
    size_t active_cores = std::thread::hardware_concurrency();
    
    // Set num threads to active_cores
    omp_set_dynamic(0);
    omp_set_num_threads(active_cores);
    
    log_info() << "HMMTrainer initialized: Thread pool set to " << omp_get_max_threads();
    log_info() << "HMMTrainer initialized: Batch size set to " << max_batch_files_;
}

void HMMTrainer::initialize() {
    size_t n_states = hmm_.get_n_states();
    HMMSVInitAccumulator global_acc = hmm_.create_init_accumulator();
    
    // Initialize useful variables for the loop
    size_t i = 0;
    size_t total_files = data_loader_.get_num_files();
    
    // Useful vectors containing batch and accumulators
    std::vector<SequenceAndLabel> batch_files;
    std::vector<HMMSVInitAccumulator> local_accs;
    for (size_t i = 0; i < max_batch_files_; ++i)
        local_accs.push_back(hmm_.create_init_accumulator());
    
    for (auto&& pair : data_loader_.stream_sequences_and_labels()) {
        // increment i first so modulo batch_size doesn't include the empty batch
        if (batch_files.size() == max_batch_files_)
            batch_files[i % max_batch_files_] = std::move(pair);
        else
            batch_files.push_back(std::move(pair));
        ++i;
        
        size_t i_mod_max = i % max_batch_files_;
        if (i_mod_max == 0 || i == total_files) {
            // if i % max_batch_files_ is not 0 but i == total_files this is the last batch,
            // so use the modulo. otherwise use max_batch_files_
            size_t batch_size = i_mod_max != 0 ? i_mod_max : max_batch_files_;
            
            size_t j;

            #pragma omp for schedule(static)
            for (j = 0; j < batch_size; ++j) {
                const SequenceAndLabel batch_pair = batch_files[j];
                
                const Eigen::MatrixXd& sequence = batch_pair.data;
                const std::vector<std::string>& labels = batch_pair.labels;
                
                local_accs[j].clear();
                try {
                    hmm_.init_accumulate(sequence, labels, local_accs[j]);
                } catch (std::invalid_argument e) {
                    log_error() << e.what();
                }
            }
            
            // update global acc
            for (j = 0; j < batch_size; ++j)
                global_acc += local_accs[j];
        }
    }

    log_debug() << "Running init_summarize";
    hmm_.init_summarize(global_acc);
    log_debug() << "Finished init_summarize";
}

void HMMTrainer::train_step() {
    size_t n_states = hmm_.get_n_states();
    
    // Initialize useful variables for the loop
    size_t total_files = data_loader_.get_num_files();
    
    // global accumulator for parallelized loop below.
    // accumulates in pragma omp critical.
    log_debug() << "Creating global acc\n";
    HMMSVBaumWelchAccumulator global_acc = hmm_.create_baum_welch_accumulator();
    log_debug() << "Created global acc\n";

    // Useful vectors containing batch and accumulators
    std::vector<Sequence> batch_files;
    std::vector<HMMSVBaumWelchAccumulator> local_accs;
    for (size_t i = 0; i < max_batch_files_; ++i) {
        local_accs.push_back(hmm_.create_baum_welch_accumulator());
        log_debug() << "Created local accs\n";
    }

    log_debug() << "Init HMM: \n" << hmm_;
    size_t i = 0;
    for (Sequence sequence : data_loader_.stream_sequences()) {
        if (batch_files.size() == max_batch_files_)
            batch_files[i % max_batch_files_] = std::move(sequence);
        else
            batch_files.push_back(std::move(sequence));
        ++i;
        
        size_t i_mod_max = i % max_batch_files_;
        if (i_mod_max == 0 || i == total_files) {
            // if i % max_batch_files_ is not 0 but i == total_files this is the last batch,
            // so use the modulo. otherwise use max_batch_files_
            size_t batch_size = i_mod_max != 0 ? i_mod_max : max_batch_files_;
            
            // parallelized training over the batch
            size_t j;
            #pragma omp for schedule(static)
            for (j = 0; j < batch_size; ++j) {
                const Eigen::MatrixXd& sequence_data = batch_files[j].data;
                log_debug() << "clear batch " << j << " acc.\n";
                local_accs[j].clear();
                log_debug() << "accumulating batch " << j << ".\n";
                hmm_.baum_welch_accumulate(sequence_data, local_accs[j]);
            }
            
            // update global acc
            for (j = 0; j < batch_size; ++j)
                global_acc += local_accs[j];
        }
    }

    log_debug() << "Running baum_welch_summarize";
    hmm_.baum_welch_summarize(global_acc);
    log_debug() << "Finished baum_welch_summarize";

    log_debug() << "Trained HMM: \n"<< hmm_;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

