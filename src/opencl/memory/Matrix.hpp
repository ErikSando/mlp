#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

#include "CL/cl.h"

namespace mlp {
    namespace opencl {
        class Matrix {
            public:

            Matrix(const cl_context& context, size_t rows, size_t columns);
            ~Matrix();

            Matrix(const Matrix&) = delete;
            Matrix& operator=(const Matrix&) = delete;

            Matrix(Matrix&&) noexcept;
            Matrix& operator=(Matrix&&) noexcept;

            void zero(const cl_command_queue& command_queue);

            cl_mem& data() { return m_data; }
            const cl_mem& data() const { return m_data; }

            constexpr size_t rows() const { return m_rows; }
            constexpr size_t columns() const { return m_cols; }
            constexpr size_t size() const { return m_size; }

            private:

            size_t m_rows;
            size_t m_cols;
            size_t m_size;

            cl_mem m_data;
        };
    }
}