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

    Matrix::Matrix(Matrix&& other) noexcept
    : m_size(other.m_size), m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data)
    {
        other.m_data = nullptr;
    }

    Matrix& Matrix::operator=(Matrix&& other) noexcept {
        if (this != &other) {
            cudaFree(m_data);

            m_size = other.m_size;
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = other.m_data;

            other.m_data = nullptr;
        }

        return *this;
    }
}