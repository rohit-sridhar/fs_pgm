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
 * load_file loads a single file for a single iteration of
 * load_dir.
 */

namespace fs = std::filesystem;

class DataLoader {
private:
    fs::path load_dir;
public:
    DataLoader(const fs::path& load_dir) : load_dir(load_dir) {}
    inline static const std::string DATAFILE_EXTENSION = ".dat";

    Eigen::MatrixXd load_file(const fs::path& fpath);
    std::generator<Eigen::MatrixXd> stream_sequences();

    const fs::path& get_load_dir() const { return load_dir; };
};

std::ostream& operator<<(std::ostream& os, const DataLoader& data_loader);

