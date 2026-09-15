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

    
    // import constants
    inline const double FLOOR_VAL = 1e-12;
    
    // print functions
    template <typename Scalar, int NumIndices>
    void print_dims(const Eigen::Tensor<Scalar, NumIndices>& tensor) {
        const auto& dims = tensor.dimensions();
        int rank = dims.size();
        
        std::cout << "Tensor Shape [Rank " << rank << "]: (";
        for (int i = 0; i < NumIndices; ++i) {
            std::cout << dims[i] << (i < NumIndices - 1 ? ", " : "");
        }
        std::cout << ")\n";
    }
    
    void print_dims(const Eigen::VectorXd& vec);
    void print_dims(const Eigen::MatrixXd& mat);
}

