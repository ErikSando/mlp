#pragma once

#include <cstddef>
#include <vector>

#include "matrix/Matrix.hpp"

namespace mlp {
    class HostMatrix {
        public:

        HostMatrix(size_t rows, size_t columns) : m_rows(rows), m_cols(columns), m_data(rows, columns) {}
        ~HostMatrix();

        void zero();

        float* data() { return m_data.data(); }
        const float* data() const { return m_data.data(); }

        constexpr size_t rows() const { return m_rows; }
        constexpr size_t columns() const { return m_cols; }
        constexpr size_t size() const { return m_size; }

        private:

        size_t m_rows;
        size_t m_cols;
        size_t m_size;

        std::vector<float> m_data;
    };
}