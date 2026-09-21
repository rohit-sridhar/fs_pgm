#include "data_loader.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <generator>

#include <Eigen/Dense>

#include "utils.h"

namespace fs = std::filesystem;

Eigen::MatrixXd DataLoader::load_file(const fs::path& fpath) {
    log_debug() << "Loading: " << fpath.filename();
    std::ifstream file(fpath); 
    
    if (!file.is_open()) {
        log_error() << "Error: Could not open file at " << fpath;
        return Eigen::MatrixXd();
    }

    std::string line;
    std::vector<double> flat_data;

    size_t num_rows = 0;
    size_t num_cols = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue; 

        std::stringstream ss(line);
        double value;
        size_t current_row_cols = 0;

        while (ss >> value) {
            flat_data.push_back(value);
            current_row_cols++;
        }

        if (num_rows == 0) {
            num_cols = current_row_cols;
        } else if (current_row_cols != num_cols) {
            log_error() << "Error: Row " << num_rows << " has mismatched dimensions!";
        }
        
        num_rows++;
    }
    file.close();

    // Map and copy into an Eigen Matrix
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        matrix_view(flat_data.data(), num_rows, num_cols);
        
    Eigen::MatrixXd final_matrix = matrix_view;

    log_debug() << "Successfully loaded matrix from: " << fpath.filename();
    log_debug() << "Dimensions: " << final_matrix.rows() << "x" << final_matrix.cols();

    return final_matrix;
}

std::generator<Eigen::MatrixXd> DataLoader::stream_sequences() {
    try {
        log_debug() << "fs exists: " << fs::exists(load_dir);
        log_debug() << "is_directory: " << is_directory(load_dir);
        if (fs::exists(load_dir) && fs::is_directory(load_dir)) {
            // Loop through all entries in the directory
            for (const auto& file : fs::directory_iterator(load_dir)) {
                fs::path fpath = file.path();
                fs::path fext = fpath.extension();
                // Check if it is a regular file (skips folders/symlinks if desired)
                if (fs::is_regular_file(file) && fext == DATAFILE_EXTENSION) {
                    co_yield load_file(fpath);
                }
            }
        } else {
            log_warning() << "Provided path is not a valid directory.";
        }
    } catch (const fs::filesystem_error& err) {
        log_error() << "Filesystem error: " << err.what();
    }
}

std::ostream& operator<<(std::ostream& os, const DataLoader& data_loader) {
    os << "Data Loader Load dir: " << data_loader.get_load_dir() << "\n";
    return os;
}
