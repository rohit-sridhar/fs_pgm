#include "trainer.h"

#include "hmm.h"
#include "data_loader.h"
#include "utils.h"

void HMMTrainer::train_iters(size_t iters) {
    log_info() << "train iters: " << iters;
}

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer) {
    os << hmm_trainer.get_hmm();
    os << hmm_trainer.get_data_loader();
    return os;
}

