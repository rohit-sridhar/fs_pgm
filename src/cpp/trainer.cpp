#include "trainer.h"

#include <Eigen/Dense>
#include <omp.h>
#include <thread>

#include "hmm.h"
#include "data_loader.h"
#include "utils.h"

HMMTrainer::HMMTrainer(
    HMM& hmm,
    DataLoader& data_loader,
    size_t batch_size
) : hmm(hmm), data_loader(data_loader), max_batch_files(batch_size) {
    // Query OS-allocated core count
    size_t active_cores = std::thread::hardware_concurrency();
    
    // Set num threads to active_cores
    omp_set_dynamic(0);
    omp_set_num_threads(active_cores);
    
    log_info() << "HMMTrainer initialized: Thread pool set to " << omp_get_max_threads();
    log_info() << "HMMTrainer initialized: Batch size set to " << max_batch_files;
}

void HMMTrainer::initialize() {
    size_t n_states = hmm.get_n_states();
    HMMInitAccumulator global_acc = HMMInitAccumulator(n_states);
    
    // Initialize useful variables for the loop
    size_t i = 0;
    size_t total_files = data_loader.get_num_files();
    
    // Useful vectors containing batch and accumulators
    std::vector<SequenceAndLabel> batch_files;
    std::vector<HMMInitAccumulator> local_accs(max_batch_files, HMMInitAccumulator(n_states));

    for (auto&& pair : data_loader.stream_sequences_and_labels()) {
        // increment i first so modulo batch_size doesn't include the empty batch
        if (batch_files.size() == max_batch_files) {
            batch_files[i % max_batch_files] = std::move(pair);
        } else{
            batch_files.push_back(std::move(pair));
        }
        ++i;
        
        size_t i_mod_max = i % max_batch_files;
        if (i_mod_max == 0 || i == total_files) {
            // if i % max_batch_files is not 0 but i == total_files this is the last batch,
            // so use the modulo. otherwise use max_batch_files
            size_t batch_size = i_mod_max != 0 ? i_mod_max : max_batch_files;
            
            size_t j;
            #pragma omp for schedule(static)
            for (j = 0; j < batch_size; ++j) {
                const SequenceAndLabel batch_pair = batch_files[j];
                
                const Eigen::MatrixXd& sequence = batch_pair.data;
                const std::vector<std::string>& labels = batch_pair.labels;
                
                local_accs[j].clear();
                hmm.init_accumulate(sequence, labels, local_accs[j]);
            }
            
            // update global acc
            for (j = 0; j < batch_size; ++j) {
                global_acc += local_accs[j];
            }
        }
    }

    log_debug() << "Running init_summarize";
    hmm.init_summarize(global_acc);
    log_debug() << "Finished init_summarize";
}

void HMMTrainer::train_step() {
    size_t n_states = hmm.get_n_states();
    
    // Initialize useful variables for the loop
    size_t total_files = data_loader.get_num_files();
    
    // global accumulator for parallelized loop below.
    // accumulates in pragma omp critical.
    HMMBaumWelchAccumulator global_acc = HMMBaumWelchAccumulator(n_states);
    // Useful vectors containing batch and accumulators
    std::vector<Sequence> batch_files;
    std::vector<HMMBaumWelchAccumulator> local_accs(max_batch_files, HMMBaumWelchAccumulator(n_states));

    log_debug() << "Init HMM: \n" << hmm;
    size_t i = 0;
    for (Sequence sequence : data_loader.stream_sequences()) {
        if (batch_files.size() == max_batch_files) {
            batch_files[i % max_batch_files] = std::move(sequence);
        } else {
            batch_files.push_back(std::move(sequence));
        }
        ++i;
        
        size_t i_mod_max = i % max_batch_files;
        if (i_mod_max == 0 || i == total_files) {
            // if i % max_batch_files is not 0 but i == total_files this is the last batch,
            // so use the modulo. otherwise use max_batch_files
            size_t batch_size = i_mod_max != 0 ? i_mod_max : max_batch_files;
            
            // parallelized training over the batch
            size_t j;
            #pragma omp for schedule(static)
            for (j = 0; j < batch_size; ++j) {
                const Eigen::MatrixXd& sequence_data = batch_files[j].data;
                local_accs[j].clear();
                hmm.baum_welch_accumulate(sequence_data, local_accs[j]);
            }
            
            // update global acc
            for (j = 0; j < batch_size; ++j) {
                global_acc += local_accs[j];
            }
        }
    }

    log_debug() << "Running baum_welch_summarize";
    hmm.baum_welch_summarize(global_acc);
    log_debug() << "Finished baum_welch_summarize";

    log_debug() << "Trained HMM: \n"<< hmm;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

