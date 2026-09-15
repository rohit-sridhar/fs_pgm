#include <iostream>
#include <sstream>
#include <streambuf>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "utils.h"

namespace utils {
    /** LogStream member functions **/

    // Log Stream destructor outputs to python logging
    // callback
    LogStream::~LogStream() {
        if (cb) {
            cb(oss.str());
        }
    }

    // support for stream manipulators (from iomanip/iostream)
    LogStream& LogStream::operator<<(std::ostream& (*manip)(std::ostream&)) {
        manip(oss);
        return *this;
    }

    /** CppLogger functions **/

    // These lambdas print to std cout in case they are not redefined
    // using the setter functions below. DEBUG is left empty, since 
    // we won't be able to turn these off (for now).
    CppLogger::LogCallback CppLogger::debug = [](std::string msg) {};
    CppLogger::LogCallback CppLogger::info = [](std::string msg) {
        std::cout <<  "[C++ INFO]: " << msg << std::endl;
    };
    CppLogger::LogCallback CppLogger::warning = [](std::string msg) {
        std::cout <<  "[C++ WARNING]: " << msg << std::endl;
    };
    CppLogger::LogCallback CppLogger::error = [](std::string msg) {
        std::cout <<  "[C++ ERROR]: " << msg << std::endl;
    };

    // These set the debug,info,warning, and error loggers.
    // The function passed in should log an std::string arg
    // to the appropriate channel. these are expected to come
    // from python's logging module.
    void CppLogger::set_debug_cb(CppLogger::LogCallback debug_cb) {
        CppLogger::debug = debug_cb;
    }
    void CppLogger::set_info_cb(CppLogger::LogCallback info_cb) {
        CppLogger::info = info_cb;
    }
    void CppLogger::set_warning_cb(CppLogger::LogCallback warning_cb) {
        CppLogger::warning = warning_cb;
    }
    void CppLogger::set_error_cb(CppLogger::LogCallback error_cb) {
        CppLogger::error = error_cb;
    }

    //
    LogStream CppLogger::log_debug() {
        return LogStream(debug);
    }
    LogStream CppLogger::log_info() {
        return LogStream(info);
    }
    LogStream CppLogger::log_warning() {
        return LogStream(warning);
    }
    LogStream CppLogger::log_error(){
        return LogStream(error);
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
