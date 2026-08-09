#include <iostream>

#include <cuda_runtime.h>

#include "cuda/matrix/Matrix.hpp"

namespace mlp {
    namespace cuda {
        Matrix::Matrix(const size_t rows, const size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
            cudaError_t err = cudaMalloc((void**) &m_data, m_size * sizeof(float));
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        }

        Matrix::~Matrix() {
            cudaFree(m_data);
        }

        void Matrix::zero() {
            cudaError_t err = cudaMemset(m_data, 0, m_size * sizeof(float));
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}