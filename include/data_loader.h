#pragma once

#include <filesystem>
#include <functional>
#include <generator>
#include <string>

#include <Eigen/Dense>

/*
 * DataLoader class to be used with hmm.cpp.
 *
 * The constructor saves a load directory
 *
 * stream_sequences is a generator function that
 * allows the caller to loop over the files in load_dir
 * under the assumption that each file in load_dir contains
 * a sequence of data points.
 * 
 * load_sequence_file loads a single file for a single iteration of
 * load_dir.
 */

namespace fs = std::filesystem;

struct Sequence {
    Eigen::MatrixXd data;
};

struct SequenceAndLabel {
    Eigen::MatrixXd sequence;
    std::vector<std::string> labels;
};

class DataLoader {
private:
    fs::path load_dir;
    std::vector<std::string> load_label_file(const fs::path& fpath);
    Eigen::MatrixXd load_sequence_file(const fs::path& fpath);

public:
    DataLoader(const fs::path& load_dir) : load_dir(load_dir) {}

    inline static const std::string DATAFILE_EXTENSION = ".dat";
    inline static const std::string LABFILE_EXTENSION = ".lab";
    
    std::generator<Sequence> stream_sequences();
    std::generator<SequenceAndLabel> stream_sequences_and_labels();
    
    const fs::path& get_load_dir() const { return load_dir; };
};

std::ostream& operator<<(std::ostream& os, const DataLoader& data_loader);

