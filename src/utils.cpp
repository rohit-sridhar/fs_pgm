#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils.h"

namespace utils {
    void print_dims(const Eigen::VectorXd& vec) {
        std::cout << "Tensor Shape [Rank 1]: (" << vec.size() << "," << ")\n";
    }

    void print_dims(const Eigen::MatrixXd& mat) {
        std::cout << "Tensor Shape [Rank 2]: (" << mat.rows() << ", " << mat.cols() << ")\n";
    }
}
