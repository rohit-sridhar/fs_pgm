#include "trainer.h"

#include <Eigen/Dense>
#include <omp.h>
#include <thread>

#include "hmm.h"
#include "data_loader.h"
#include "utils.h"

HMMTrainer::HMMTrainer(HMM& hmm, DataLoader& data_loader) : hmm(hmm), data_loader(data_loader) {
    // Query true OS-allocated core count
    size_t active_cores = std::thread::hardware_concurrency();

    // Set num threads to active_cores
    omp_set_dynamic(0);
    omp_set_num_threads(active_cores);

    log_info() << "HMMTrainer initialized: Thread pool set to " << active_cores;
}

void HMMTrainer::initialize() {
    size_t n_states = hmm.get_n_states();
    size_t num_cores = std::thread::hardware_concurrency();
    HMMInitAccumulator global_acc = HMMInitAccumulator(n_states);
    
    // Initialize useful variables for the loop
    size_t i = 0;
    size_t max_batch_files = 64;
    size_t total_files = data_loader.get_num_files();

    // Useful vectors containing batch and accumulators
    std::vector<SequenceAndLabel> batch_files(max_batch_files);
    std::vector<HMMInitAccumulator> local_accs(max_batch_files, HMMInitAccumulator(n_states));

    for (SequenceAndLabel pair : data_loader.stream_sequences_and_labels()) {
        // increment i first so modulo batch_size doesn't include the empty batch
        batch_files[i % max_batch_files] = pair;
        ++i;
        
        if (i % max_batch_files == 0 || i == total_files) {
            size_t batch_size = i == total_files ? max_batch_files : total_files;
            
            log_info() << "Before pragma omp parallel";
            #pragma omp parallel for
            for (size_t j = 0; j < batch_size; ++j) {
                SequenceAndLabel batch_pair = batch_files[j];
                
                log_info() << "Running init_accumulate ...";
                Eigen::MatrixXd sequence = batch_pair.sequence;
                std::vector<std::string> labels = batch_pair.labels;
                
                hmm.init_accumulate(sequence, labels, local_accs[j]);
            }
            log_info() << "After pragma omp parallel";
            
            // update global acc
            for (size_t j = 0; j < batch_size; ++j) {
                global_acc += local_accs[j];
            }
        }
    }

    log_debug() << "Running init_summarize";
    hmm.init_summarize(global_acc);
    log_debug() << "Finished init_summarize";
}

void HMMTrainer::train(size_t iters) {
    size_t n_states = hmm.get_n_states();
    size_t num_cores = std::thread::hardware_concurrency();
    log_debug() << "Init HMM: \n" << hmm;

    for (size_t i = 0; i < iters; ++i) {
        if (i % 5 == 0) {
            log_info() << "Iter: " << i;
        }
        
        // global accumulator for parallelized loop below.
        // accumulates in pragma omp critical.
        HMMBaumWelchAccumulator global_acc = HMMBaumWelchAccumulator(n_states);
        
        log_info() << "Before pragma omp parallel";
        // initialize thread pool (parallelization block)
        #pragma omp parallel shared(global_acc)
        {
            log_info() << "Inside pragma omp parallel";
            #pragma omp single
            {
                log_info() << "Spawning " << omp_get_num_threads() << " threads for training";
                for (Sequence sequence : data_loader.stream_sequences()) {
                    #pragma omp task firstprivate(sequence)
                    {
                        HMMBaumWelchAccumulator local_acc = HMMBaumWelchAccumulator(n_states);
                        hmm.baum_welch_accumulate(sequence.data, local_acc);
                        
                        #pragma omp critical
                        {
                            global_acc += local_acc;
                        }
                    }
                }
            }
        }
        
        hmm.baum_welch_summarize(global_acc);
        log_debug() << hmm << "\n";
    }
    log_debug() << "Trained HMM: \n"<< hmm;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

