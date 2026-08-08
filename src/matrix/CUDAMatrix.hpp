#pragma once

#include <cstddef>

#include "matrix/Matrix.hpp"

namespace mlp {
    // A wrapper for a float array stored on device memory
    class CUDAMatrix : public IMatrix {
        public:

        CUDAMatrix(size_t rows, size_t columns);
        ~CUDAMatrix();

        CUDAMatrix(const CUDAMatrix&) = delete;
        CUDAMatrix& operator=(const CUDAMatrix&) = delete;

        CUDAMatrix(CUDAMatrix&& other) noexcept;
        CUDAMatrix& operator=(CUDAMatrix&& other) noexcept;

        void zero() override;

        float* data() override { return m_data; } // device memory address of the matrix values
        const float* data() const override { return m_data; } // device memory address of the matrix values

        private:

        float* m_data = nullptr;
    };
}