#pragma once

#include <cstddef>

#include "matrix/Matrix.hpp"

namespace mlp {
    // A wrapper for a float array stored on device memory
    class CUDAMatrix {
        public:

        CUDAMatrix(size_t rows, size_t columns);
        ~CUDAMatrix();

        CUDAMatrix(const CUDAMatrix&) = delete;
        CUDAMatrix& operator=(const CUDAMatrix&) = delete;

        CUDAMatrix(CUDAMatrix&& other) noexcept;
        CUDAMatrix& operator=(CUDAMatrix&& other) noexcept;

        void zero();

        float* data() { return m_data; } // device memory address of the matrix values
        const float* data() const { return m_data; } // device memory address of the matrix values

        constexpr size_t rows() const { return m_rows; }
        constexpr size_t columns() const { return m_cols; }
        constexpr size_t size() const { return m_size; }

        private:

        size_t m_rows;
        size_t m_cols;
        size_t m_size;

        float* m_data = nullptr;
    };
}