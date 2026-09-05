#include <iostream>

#include "opencl/memory/Matrix.hpp"

namespace mlp {
    namespace opencl {
        Matrix::Matrix(const size_t rows, const size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
            
        }

        Matrix::~Matrix() {
            if (m_data) clReleaseMemObject(m_data);
        }

        void Matrix::zero() {
            
        }
    }
}