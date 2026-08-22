#include <iostream>

#include <cuda_runtime.h>

#include "cuda/memory/Matrix.hpp"

namespace mlp {
    namespace cuda {
        Matrix::Matrix(const size_t rows, const size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
            cudaError_t err = cudaMalloc((void**) &m_data, m_size * sizeof(float));
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        }

        Matrix::~Matrix() {
            cudaFree(m_data);
        }

        Matrix::Matrix(Matrix&& other) noexcept
        : m_rows(other.m_rows), m_cols(other.m_cols), m_size(other.m_size), m_data(other.m_data)
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

                m_data = other.m_data;
                other.m_data = nullptr;
            }

            return *this;
        }

        void Matrix::zero() {
            cudaError_t err = cudaMemset(m_data, 0, m_size * sizeof(float));
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}