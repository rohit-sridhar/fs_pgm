#include "trainer.h"

#include <Eigen/Dense>
#include <omp.h>
#include <thread>

#include "hmm.h"
#include "data_loader.h"
#include "utils.h"

void HMMTrainer::initialize() {
    size_t n_states = hmm.get_n_states();
    size_t num_cores = std::thread::hardware_concurrency();
    HMMInitAccumulator global_acc = HMMInitAccumulator(n_states);
    log_info() << "num threads: " << num_cores;

    // initialize thread pool (parallelization block)
    #pragma omp parallel num_threads(num_cores)
    {
        // have main thread start for loop (generator size is unknown)
        #pragma omp single
        {
            for (SequenceAndLabel pair : data_loader.stream_sequences_and_labels()) {
                // assign thread from thread pool for inner loop code
                #pragma omp task firstprivate(pair)
                {
                    HMMInitAccumulator local_acc = HMMInitAccumulator(n_states);
                    Eigen::MatrixXd sequence = pair.sequence;
                    std::vector<std::string> labels = pair.labels;
                    hmm.init_accumulate(sequence, labels, local_acc);

                    #pragma omp critical
                    {
                        global_acc += local_acc;
                    }
                }
            }
        }
    }

    hmm.init_summarize(global_acc);
}

void HMMTrainer::train(size_t iters) {
    size_t n_states = hmm.get_n_states();
    size_t num_cores = std::thread::hardware_concurrency();
    log_debug() << "Init HMM: \n" << hmm;

    for (size_t i = 0; i < iters; ++i) {
        if (i % 5 == 0) {
            log_info() << "Iter " << i;
        }

        // global accumulator for parallelized loop below.
        // accumulates in pragma omp critical.
        HMMBaumWelchAccumulator global_acc = HMMBaumWelchAccumulator(n_states);

        // initialize thread pool (parallelization block)
        #pragma omp parallel num_threads(num_cores)
        {
            // have main thread start for loop (generator size is unknown)
            #pragma omp single
            {
                for (Sequence sequence : data_loader.stream_sequences()) {
                    HMMBaumWelchAccumulator local_acc = HMMBaumWelchAccumulator(n_states);
                    hmm.baum_welch_accumulate(sequence.data, local_acc);
                    
                    #pragma omp critical
                    {
                        global_acc += local_acc;
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

