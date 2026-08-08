#include <iostream>

#include <cuda_runtime.h>

#include "matrix/CUDAMatrix.hpp"
#include "matrix/Matrix.hpp"

namespace mlp {
    DeviceInt::DeviceInt(const int value) {
        cudaError_t err = cudaMalloc((void**) &m_data, sizeof(int));
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

        err = cudaMemcpy((void*) m_data, (void*) &value, sizeof(int), cudaMemcpyHostToDevice);
        // if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");
    }

    CUDAMatrix::CUDAMatrix(const size_t rows, const size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
        cudaError_t err = cudaMalloc((void**) &m_data, m_size * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }

    CUDAMatrix::~CUDAMatrix() {
        cudaFree(m_data);
    }

    CUDAMatrix::CUDAMatrix(CUDAMatrix&& other) noexcept
    : m_rows(other.m_rows), m_cols(other.m_cols), m_size(other.m_size), m_data(other.m_data)
    {
        other.m_data = nullptr;
    }

    CUDAMatrix& CUDAMatrix::operator=(CUDAMatrix&& other) noexcept {
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

    void CUDAMatrix::zero() {
        cudaError_t err = cudaMemset(m_data, 0, m_size * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
}