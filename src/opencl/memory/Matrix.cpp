#include <iostream>

#include "opencl/CLstuff.hpp"
#include "opencl/memory/Matrix.hpp"

namespace mlp {
    namespace opencl {
        Matrix::Matrix(const size_t rows, const size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
            if (m_size == 0) return;

            cl_int err;

            cl_mem_flags mem_flags = CL_MEM_READ_WRITE;

            m_data = clCreateBuffer(clcontext, mem_flags, m_size * sizeof(float), nullptr, &err);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to create device buffer");
            }

            if (!m_data) {
                ERROR("Failed to create device buffer");
            }
        }

        Matrix::~Matrix() {
            if (m_data) clReleaseMemObject(m_data);
        }

        void Matrix::zero() {
        }
    }
}