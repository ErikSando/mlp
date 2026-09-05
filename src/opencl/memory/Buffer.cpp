#include <iostream>

#include "opencl/memory/Buffer.hpp"

namespace mlp {
    namespace opencl {
        Buffer::Buffer(const size_t size) : m_size(size) {
            cl_int err;

            cl_mem_flags mem_flags = CL_MEM_READ_WRITE;

            // m_data = clCreateBuffer()
        }

        Buffer::~Buffer() {
            if (m_data) clReleaseMemObject(m_data);
        }

        void Buffer::zero() {
            
        }
    }
}