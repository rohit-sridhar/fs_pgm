#include "hmm.h"

// The accumulators below work for initialization and baum welch.
// They accumulate (sums) of values in parallel and the HMM class
// aggregates (averages, or keeps the sum) them after the threads
// join. We keep these separate since each thread will work on its
// own copy.
struct HMMSVInitAccumulator : HMMInitAccumulator {
    Eigen::VectorXd means;
    Eigen::VectorXd std_devs;
    
    HMMSVInitAccumulator(size_t n_states) : HMMInitAccumulator(n_states) {
        clear();
    }

    void clear() override {
        HMMInitAccumulator::clear();

        means.setZero(n_states);
        std_devs.setZero(n_states);
    }

    HMMSVInitAccumulator& operator+=(const HMMSVInitAccumulator& acc) {
        HMMInitAccumulator::operator+=(acc);

        means += acc.means;
        std_devs += acc.std_devs;

        return *this;
    }
};

struct HMMSVBaumWelchAccumulator : HMMBaumWelchAccumulator {
    Eigen::VectorXd obs;
    Eigen::VectorXd obs_sq;
    
    HMMSVBaumWelchAccumulator(size_t n_states) : HMMBaumWelchAccumulator(n_states) {
        clear();
    }
    
    void clear() {
        HMMBaumWelchAccumulator::clear();
        
        obs.setZero(n_states);
        obs_sq.setZero(n_states);
    }
    
    HMMSVBaumWelchAccumulator& operator+=(const HMMSVBaumWelchAccumulator& acc) {
        HMMBaumWelchAccumulator::operator+=(acc);
        
        obs += acc.obs;
        obs_sq += acc.obs_sq;
        
        return *this;
    }
};


