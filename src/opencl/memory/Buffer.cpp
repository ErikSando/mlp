#include <iostream>

#include "opencl/ContextImpl.hpp"
#include "opencl/memory/Buffer.hpp"

namespace mlp {
    namespace opencl {
        Buffer::Buffer(const cl_context& context, const size_t size) : m_size(size) {
            if (m_size == 0) return;

            cl_int err;

            cl_mem_flags mem_flags = CL_MEM_READ_WRITE;

            m_data = clCreateBuffer(context, mem_flags, m_size, nullptr, &err);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to create device buffer");
            }

            if (!m_data) {
                ERROR("Failed to create device buffer");
            }
        }

        Buffer::~Buffer() {
            if (m_data) clReleaseMemObject(m_data);
        }

        void Buffer::zero(const cl_command_queue& command_queue) {
            const uint8_t zero = 0; // 1 byte
            cl_int err = clEnqueueFillBuffer(command_queue, m_data, (void*) &zero, sizeof(zero), 0, m_size, 0, nullptr, nullptr);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to zero device buffer");
            }
        }
    }
}