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
    std::vector<SequenceAndLabel> batch_files(max_batch_files);
    std::vector<HMMInitAccumulator> local_accs(max_batch_files, HMMInitAccumulator(n_states));

    #pragma omp parallel
    {
        for (SequenceAndLabel pair : data_loader.stream_sequences_and_labels()) {
            // increment i first so modulo batch_size doesn't include the empty batch
            batch_files[i % max_batch_files] = pair;
            ++i;
            
            size_t i_mod_max = i % max_batch_files;
            if (i_mod_max == 0 || i == total_files) {
                // if i % max_batch_files is not 0 but i == total_files this is the last batch,
                // so use the modulo. otherwise use max_batch_files
                size_t batch_size = i_mod_max != 0 ? i_mod_max : max_batch_files;
                
                #pragma omp for schedule(static)
                for (size_t j = 0; j < batch_size; ++j) {
                    const SequenceAndLabel& batch_pair = batch_files[j];
                    
                    // log_info() << "Running init_accumulate ...";
                    const Eigen::MatrixXd& sequence = batch_pair.sequence;
                    const std::vector<std::string>& labels = batch_pair.labels;
                    
                    local_accs[j].clear();
                    hmm.init_accumulate(sequence, labels, local_accs[j]);
                }
                
                // update global acc
                for (size_t j = 0; j < batch_size; ++j) {
                    global_acc += local_accs[j];
                }
            }
        }
    }

    log_debug() << "Running init_summarize";
    hmm.init_summarize(global_acc);
    log_debug() << "Finished init_summarize";
}

void HMMTrainer::train(size_t iters) {
    size_t n_states = hmm.get_n_states();
    
    // Initialize useful variables for the loop
    size_t total_files = data_loader.get_num_files();
    
    // global accumulator for parallelized loop below.
    // accumulates in pragma omp critical.
    HMMBaumWelchAccumulator global_acc = HMMBaumWelchAccumulator(n_states);
    // Useful vectors containing batch and accumulators
    std::vector<Sequence> batch_files(max_batch_files);
    std::vector<HMMBaumWelchAccumulator> local_accs(max_batch_files, HMMBaumWelchAccumulator(n_states));

    log_debug() << "Init HMM: \n" << hmm;
    #pragma omp parallel
    {
        for (size_t i = 0; i < iters; ++i) {
            if (i % 5 == 0) {
                log_info() << "Iter: " << i;
            }
            
            global_acc.clear();
            size_t j = 0;

            for (Sequence sequence : data_loader.stream_sequences()) {
                batch_files[j % max_batch_files] = sequence;
                ++j;
                
                size_t j_mod_max = j % max_batch_files;
                if (j_mod_max == 0 || j == total_files) {
                    // if j % max_batch_files is not 0 but j == total_files this is the last batch,
                    // so use the modulo. otherwise use max_batch_files
                    size_t batch_size = j_mod_max != 0 ? j_mod_max : max_batch_files;
                    
                    // parallelized training over the batch
                    #pragma omp for schedule(static)
                    for (size_t k = 0; k < batch_size; ++k) {
                        const Eigen::MatrixXd& sequence_data = batch_files[k].data;
                        local_accs[k].clear();
                        hmm.baum_welch_accumulate(sequence_data, local_accs[k]);
                    }
                    
                    // update global acc
                    for (size_t k = 0; k < batch_size; ++k) {
                        global_acc += local_accs[k];
                    }
                }
            }

            log_debug() << "Running baum_welch_summarize";
            hmm.baum_welch_summarize(global_acc);
            log_debug() << "Finished baum_welch_summarize";
        }
    }
    log_debug() << "Trained HMM: \n"<< hmm;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

