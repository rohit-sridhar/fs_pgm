#include <iostream>
#include <streambuf>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "utils.h"

namespace utils {
    // initialize these to empty lambdas in case
    CppLogger::LogCallback CppLogger::debug = [](std::string msg) {
        std::cout <<  "[C++ DEBUG]: " << msg << std::endl;
    };
    CppLogger::LogCallback CppLogger::info = [](std::string msg) {
        std::cout <<  "[C++ INFO]: " << msg << std::endl;
    };
    CppLogger::LogCallback CppLogger::warning = [](std::string msg) {
        std::cout <<  "[C++ WARNING]: " << msg << std::endl;
    };
    CppLogger::LogCallback CppLogger::error = [](std::string msg) {
        std::cout <<  "[C++ ERROR]: " << msg << std::endl;
    };

    void CppLogger::set_debug_cb(LogCallback debug_cb) {
        CppLogger::debug = debug_cb;
    }
    void CppLogger::set_info_cb(LogCallback info_cb) {
        CppLogger::info = info_cb;
    }
    void CppLogger::set_warning_cb(LogCallback warning_cb) {
        CppLogger::warning = warning_cb;
    }
    void CppLogger::set_error_cb(LogCallback error_cb) {
        CppLogger::error = error_cb;
    }

    // print dims of an Eigen::Vector
    void print_dims(const Eigen::VectorXd& vec) {
        std::cout << "Tensor Shape [Rank 1]: (" << vec.size() << "," << ")\n";
    }

    // print dims of Eigen Matrix
    void print_dims(const Eigen::MatrixXd& mat) {
        std::cout << "Tensor Shape [Rank 2]: (" << mat.rows() << ", " << mat.cols() << ")\n";
    }

}
