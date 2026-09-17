#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <streambuf>
#include <functional>
#include <string>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace utils {
    // forward declare LogStream
    class LogStream;
    
    /*
     * CppLogger class has entirely static objs
     * Allows user to set python function handles
     * (last 4 functions are nanobound) for logging.
     *
     * These are the LogCallBack functions. To actually
     * log, messages can be streaed into the log_* static
     * funcs. These build LogStream objects based on the
     * LogCallback functions and allow << streaming into
     * the LogCallback functions for debug,info, etc.
     */
    class CppLogger {
    public:
        using LogCallback = std::function<void(std::string)>;
        
        static LogCallback debug;
        static LogCallback info;
        static LogCallback warning;
        static LogCallback error;
        
        static LogStream log_debug();
        static LogStream log_info();
        static LogStream log_warning();
        static LogStream log_error(); 
        
        static void set_debug_cb(LogCallback debug_cb);
        static void set_info_cb(LogCallback info_cb);
        static void set_warning_cb(LogCallback warning_cb);
        static void set_error_cb(LogCallback error_cb);
    };
    
    /*
     * LogStream class
     *
     * The LogStream class wraps a LogCallback function and overrides
     * the streaming operator << to allow users to stream messages
     * into the LogCallback function as they would with std::cout
     *
     * In general, the log callbacks would be set by the module calling
     * one of the cpp modules in python (so a python logging function).
     */
    class LogStream {
    public:
        CppLogger::LogCallback& cb;
        std::ostringstream oss;
        
        LogStream(CppLogger::LogCallback& callback) : cb(callback) {}
        ~LogStream();
        
        // Accept any streamable type
        template <typename T>
        LogStream& operator<<(const T& val) {
            oss << val;
            return *this;
        }
        
        LogStream& operator<<(std::ostream& (*manip)(std::ostream&));
    };

    
    /*
     * Constants
     *
     * Floor val for variance/std_dev
     */
    inline const double FLOOR_VAL = 1e-12;
    
    /*
     * Print dims of EigenTensor for any number
     * of dimensions. For now, prints to log info
     * stream. TODO: add functionality to accept
     * output stream handler as arg
     */
    template <typename Scalar, int NumIndices>
    void print_dims(const Eigen::Tensor<Scalar, NumIndices>& tensor) {
        const auto& dims = tensor.dimensions();
        int rank = dims.size();
        
        CppLogger::log_info() << "Tensor Shape [Rank " << rank << "]: (";
        for (int i = 0; i < NumIndices; ++i) {
            CppLogger::log_info() << dims[i] << (i < NumIndices - 1 ? ", " : "");
        }
        CppLogger::log_info() << ")";
    }
    
    /*
     * Print dims of Eigen::Vector
     */
    void print_dims(const Eigen::VectorXd& vec);

    /*
     * Print dims of Eigen::Matrix
     */
    void print_dims(const Eigen::MatrixXd& mat);
}

// make inline aliases for logging utilities
constexpr auto log_debug = &utils::CppLogger::log_debug;
constexpr auto log_info = &utils::CppLogger::log_info;
constexpr auto log_warning = &utils::CppLogger::log_warning;
constexpr auto log_error = &utils::CppLogger::log_error;

