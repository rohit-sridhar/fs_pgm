#pragma once

#include <iostream>

#include "hmm.h"
#include "data_loader.h"

class HMMTrainer {
public:
    HMMTrainer(HMM& hmm, DataLoader& data_loader, size_t batch_size);
    
    void initialize();
    void train_step();
    
    const HMM& get_hmm() const { return hmm_; };
    const DataLoader& get_data_loader() const { return data_loader_; };
    const size_t get_max_batch_files() const { return max_batch_files_; };
private:
    HMM& hmm_;
    DataLoader& data_loader_;
    size_t max_batch_files_;
};

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer);

