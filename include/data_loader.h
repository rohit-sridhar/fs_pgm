#pragma once

#include <iostream>
#include <filesystem>
#include <functional>
#include <generator>
#include <string>
#include <vector>

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
    std::reference_wrapper<const Eigen::MatrixXd> data;
    
    Sequence(const Eigen::MatrixXd& data) : data(data) {}
};

struct SequenceAndLabel {
    std::reference_wrapper<const Eigen::MatrixXd> data;
    std::reference_wrapper<const std::vector<std::string>> labels;

    SequenceAndLabel(
        const Eigen::MatrixXd& data,
        const std::vector<std::string>& labels
    ) : data(data), labels(labels) {}
};

class DataLoader {
public:
    DataLoader(const fs::path& load_dir);

    inline static const std::string datafile_extension = ".dat";
    inline static const std::string labfile_extension = ".lab";

    bool is_data_file(const fs::path& file) const;
    bool is_label_file(const fs::path& file) const;
    
    std::generator<Sequence&&> stream_sequences() const;
    std::generator<SequenceAndLabel&&> stream_sequences_and_labels() const;
    
    size_t get_num_files() const { return num_files; }
    const fs::path& get_load_dir() const { return load_dir; };
private:
    fs::path load_dir;
    size_t num_files;
    
    std::unordered_map<fs::path, Eigen::MatrixXd> sequences;
    std::unordered_map<fs::path, std::vector<std::string>> labels;
    
    size_t count_data_files() const;
    void load_sequences_and_labels();
    
    std::vector<std::string> load_label_file(const fs::path& fpath) const;
    Eigen::MatrixXd load_sequence_file(const fs::path& fpath) const;
};

std::ostream& operator<<(std::ostream& os, const DataLoader& data_loader);

