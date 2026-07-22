#include <iostream>

#include <cuda_runtime.h>

#include "matrix/Matrix.hpp"

namespace mlp {
    Matrix::Matrix(size_t rows, size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
        cudaError_t err = cudaMalloc((void**) &m_data, rows * columns * sizeof(float));

        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }

    Matrix::~Matrix() {
        cudaFree(m_data);
    }
}