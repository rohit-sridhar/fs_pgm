#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace utils {
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

