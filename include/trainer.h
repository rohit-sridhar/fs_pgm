#pragma once

#include "hmm.h"
#include "data_loader.h"

class HMMTrainer {
private:
    HMM hmm;
    DataLoader data_loader;
    
public:
    HMMTrainer(HMM& hmm, DataLoader& data_loader) : hmm(hmm), data_loader(data_loader) {};
    
    void train_iters(size_t iters);
    
    const HMM& get_hmm() const { return hmm; };
    const DataLoader& get_data_loader() const {return data_loader; };
};

std::ostream& operator<<(std::ostream& os, const HMMTrainer& hmm_trainer);

