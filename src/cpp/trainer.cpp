#include "trainer.h"

#include <Eigen/Dense>

#include "hmm.h"
#include "data_loader.h"
#include "utils.h"

void HMMTrainer::initialize() {
    hmm.init_reset_accumulators();
    for (SequenceAndLabel pair : data_loader.stream_sequences_and_labels()) {
        Eigen::MatrixXd sequence = pair.sequence;
        std::vector<std::string> labels = pair.labels;
        hmm.init_accumulate(sequence, labels);
    }
    hmm.init_summarize();
}

void HMMTrainer::train(size_t iters) {
    log_debug() << "Init HMM: \n" << hmm;
    for (size_t i = 0; i < iters; ++i) {
        if (i % 5 == 0) {
            log_info() << "Iter " << i;
        }
        hmm.baum_welch_reset_accumulators();
        for (Sequence sequence : data_loader.stream_sequences()) {
            hmm.baum_welch_accumulate(sequence.data);
        }
        hmm.baum_welch_summarize();
        log_debug() << hmm << "\n";
    }
    log_debug() << "Trained HMM: \n"<< hmm;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

