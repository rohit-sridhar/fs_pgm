#include "data_loader.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <generator>

#include <Eigen/Dense>

#include "utils.h"

namespace fs = std::filesystem;

/************************************************** PRIVATE FXN **************************************************/
/* load_label_file()
 *  loads a label file by reading it (taking a path to it
 *  as an arg), copying the lines into a vector and returning
 *  the vector.
 */
std::vector<std::string> DataLoader::load_label_file(const fs::path& fpath) {
    log_debug() << "Loading Label File: " << fpath.filename();
    std::ifstream file(fpath); 
    
    if (!file.is_open()) {
        log_error() << "Error: Could not open label file at " << fpath;
        return std::vector<std::string>();
    }

    std::string line;
    std::vector<std::string> labels;

    while (std::getline(file, line)) {
        if (line.empty()) {
            log_error() << "Line of label file " << fpath << " is empty.";
            log_error() << "Did not load label file.";
            return std::vector<std::string>();
        }

        // check for carriage returns (just in case)
        if (line.back() == '\r') {
            line.pop_back();
        }

        labels.push_back(line);
    }

    log_debug() << "Successfully loaded " << labels.size() << " labels from " << fpath.filename();
    return labels;
}

/*
 * load_sequence_file()
 *  Loads a sequence file by reading it. It is expected to have one frame per row
 *  and m feature columns per frame. The data is mapped into an Eigen::MatrixXd and
 *  returned.
 */
Eigen::MatrixXd DataLoader::load_sequence_file(const fs::path& fpath) {
    log_debug() << "Loading Sequence File: " << fpath.filename();
    std::ifstream file(fpath); 
    
    if (!file.is_open()) {
        log_error() << "Error: Could not open sequence file at " << fpath;
        return Eigen::MatrixXd();
    }

    std::string line;
    std::vector<double> flat_data;

    size_t num_rows = 0;
    size_t num_cols = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            log_error() << "Line of sequence file " << fpath << " is empty.";
            log_error() << "Did not load sequence file.";
            return Eigen::MatrixXd();
        }; 

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
            return Eigen::MatrixXd();
        }
        
        num_rows++;
    }

    // Map and copy into an Eigen Matrix
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        matrix_view(flat_data.data(), num_rows, num_cols);
        
    Eigen::MatrixXd final_matrix = matrix_view;

    log_debug() << "Successfully loaded matrix from: " << fpath.filename();
    log_debug() << "Dimensions: " << final_matrix.rows() << "x" << final_matrix.cols();

    return final_matrix;
}

/************************************************** PUBLIC FXN **************************************************/
/*
 * stream_sequences_and_labels()
 *  streams data and labels 1 by 1. yields a struct that wraps an
 *  EigenMatrix and std::vector<std::string>>.
 */
std::generator<SequenceAndLabel> DataLoader::stream_sequences_and_labels() {
    try {
        if (fs::exists(load_dir) && fs::is_directory(load_dir)) {
            
            // Loop through all entries in the directory
            for (const auto& file : fs::directory_iterator(load_dir)) {
                fs::path fpath = file.path();
                fs::path fext = fpath.extension();
                
                // Check if it is a regular file (skips folders/symlinks if desired)
                if (fs::is_regular_file(file) && fext == DATAFILE_EXTENSION) {
                    Eigen::MatrixXd sequence = load_sequence_file(fpath);
                    fpath.replace_extension(LABFILE_EXTENSION);
                    std::vector<std::string> labels = load_label_file(fpath);
                    co_yield {sequence, labels};
                }
            }
        } else {
            log_warning() << "Provided path is not a valid directory.";
        }
    } catch (const fs::filesystem_error& err) {
        log_error() << "Filesystem error: " << err.what();
    }
}

/*
 * stream sequences()
 *  streams data files 1 by 1. yields an EigenMatrix wrapped
 *  by a Sequence struct using the load function above.
 */
std::generator<Sequence> DataLoader::stream_sequences() {
    try {
        if (fs::exists(load_dir) && fs::is_directory(load_dir)) {
            // Loop through all entries in the directory
            for (const auto& file : fs::directory_iterator(load_dir)) {
                fs::path fpath = file.path();
                fs::path fext = fpath.extension();
                // Check if it is a regular file (skips folders/symlinks if desired)
                if (fs::is_regular_file(file) && fext == DATAFILE_EXTENSION) {
                    co_yield Sequence{load_sequence_file(fpath)};
                }
            }
        } else {
            log_warning() << "Provided path is not a valid directory.";
        }
    } catch (const fs::filesystem_error& err) {
        log_error() << "Filesystem error: " << err.what();
    }
}

/*
 * overload the stream operator to print out the DataLoader. Currently
 * only prints the load directory. TODO print out number of files as
 * well.
 */
std::ostream& operator<<(std::ostream& os, const DataLoader& data_loader) {
    os << "Data Loader Load dir: " << data_loader.get_load_dir() << "\n";
    return os;
}
