// #pragma once

// #include <cstddef>
// #include <vector>

// #include "matrix/Matrix.hpp"

// namespace mlp {
//     class HostMatrix : public Matrix {
//         public:

//         HostMatrix(size_t rows, size_t columns) : m_data(rows, columns) {}
//         ~HostMatrix();

//         const float* data() const { return m_data.data(); } // device memory address of the matrix values

//         size_t size() const { return m_size; }
//         size_t rows() const { return m_rows; }
//         size_t columns() const { return m_cols; }

//         private:

//         size_t m_size;
//         size_t m_rows;
//         size_t m_cols;

//         std::vector<float> m_data;
//     };
// }