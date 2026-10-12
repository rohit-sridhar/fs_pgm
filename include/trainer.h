#pragma once

#include "hmm.h"
#include "data_loader.h"

class HMMTrainer {
private:
    HMM& hmm;
    DataLoader& data_loader;
    size_t max_batch_files;
    
public:
    HMMTrainer(HMM& hmm, DataLoader& data_loader, size_t batch_size);
    
    void initialize();
    void train(size_t iters);
    
    const HMM& get_hmm() const { return hmm; };
    const DataLoader& get_data_loader() const { return data_loader; };
};

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer);

