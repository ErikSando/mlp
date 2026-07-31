#pragma once

#include <cstddef>

namespace mlp {
    class Matrix {
        public:

        Matrix(size_t rows, size_t columns);
        ~Matrix();

        Matrix(const Matrix&) = delete;
        Matrix& operator=(const Matrix&) = delete;

        Matrix(Matrix&& other) noexcept;
        Matrix& operator=(Matrix&& other) noexcept;

        float* data() const { return m_data; } // device memory address of the matrix values

        size_t size() const { return m_size; }
        size_t rows() const { return m_rows; }
        size_t columns() const { return m_cols; }

        private:

        size_t m_size;
        size_t m_rows;
        size_t m_cols;

        float* m_data = nullptr;
    };
}