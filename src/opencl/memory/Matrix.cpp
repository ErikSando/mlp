#include <iostream>

#include "opencl/ContextImpl.hpp"
#include "opencl/memory/Matrix.hpp"

namespace mlp {
    namespace opencl {
        Matrix::Matrix(const cl_context& context, const size_t rows, const size_t columns) : m_rows(rows), m_cols(columns), m_size(rows * columns) {
            if (m_size == 0) return;

            cl_int err;

            cl_mem_flags mem_flags = CL_MEM_READ_WRITE;

            m_data = clCreateBuffer(context, mem_flags, m_size * sizeof(float), nullptr, &err);

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

        Matrix::Matrix(Matrix&& other) noexcept
        : m_rows(other.m_rows), m_cols(other.m_cols), m_size(other.m_size), m_data(other.m_data)
        {
            other.m_data = nullptr;
        }

        Matrix& Matrix::operator=(Matrix&& other) noexcept {
            if (this != &other) {
                clReleaseMemObject(m_data);

                m_size = other.m_size;
                m_rows = other.m_rows;
                m_cols = other.m_cols;
                m_data = other.m_data;

                m_data = other.m_data;
                other.m_data = nullptr;
            }

            return *this;
        }

        void Matrix::zero(const cl_command_queue& command_queue) {
            const float zero = 0.0f;
            cl_int err = clEnqueueFillBuffer(command_queue, m_data, (void*) &zero, sizeof(zero), 0, m_size * sizeof(float), 0, nullptr, nullptr);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to zero device buffer");
            }
        }
    }
}