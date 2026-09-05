#include <iostream>

#include "opencl/CLstuff.hpp"
#include "opencl/memory/Buffer.hpp"

namespace mlp {
    namespace opencl {
        Buffer::Buffer(const size_t size) : m_size(size) {
            if (m_size == 0) return;

            cl_int err;

            cl_mem_flags mem_flags = CL_MEM_READ_WRITE;

            m_data = clCreateBuffer(clcontext, mem_flags, m_size, nullptr, &err);

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

        void Buffer::zero() {
            
        }
    }
}