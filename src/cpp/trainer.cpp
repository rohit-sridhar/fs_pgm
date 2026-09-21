#include "trainer.h"

#include <Eigen/Dense>

#include "hmm.h"
#include "data_loader.h"
#include "utils.h"

void HMMTrainer::train(size_t iters) {
    log_debug() << "Init HMM: \n" << hmm;
    for (size_t i = 0; i < iters; ++i) {
        if (i % 5 == 0) {
            log_info() << "Iter " << i;
        }
        for (Eigen::MatrixXd data : data_loader.stream_sequences()) {
            hmm.baum_welch(data);
        }
        log_debug() << hmm << "\n";
    }
    log_debug() << "Trained HMM: \n"<< hmm;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

