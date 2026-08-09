#pragma once

#include <cstddef>

namespace mlp {
    namespace cuda {
    // A wrapper for a float array stored on device memory
        class Matrix {
            public:

            Matrix(size_t rows, size_t columns);
            ~Matrix();

            // Matrix(const Matrix&) = delete;
            // Matrix& operator=(const Matrix&) = delete;

            // Matrix(Matrix&& other) noexcept;
            // Matrix& operator=(Matrix&& other) noexcept;

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
}