#pragma once

#include <cstddef>

namespace mlp {
    class Matrix {
        public:

        Matrix(size_t rows, size_t columns);
        ~Matrix();

        float* data() { return m_data; }

        size_t size() { return m_size; }
        size_t rows() { return m_rows; }
        size_t columns() { return m_cols; }

        private:

        size_t m_size;
        size_t m_rows;
        size_t m_cols;

        float* m_data;
    };
}