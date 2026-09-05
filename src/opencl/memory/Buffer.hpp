#pragma once

#include <cstddef>

#include "CL/cl.h"

namespace mlp {
    namespace opencl {
        class Buffer {
            public:

            Buffer(size_t size);
            ~Buffer();

            void zero();

            cl_mem& data() { return m_data; }
            const cl_mem& data() const { return m_data; }

            constexpr size_t size() const { return m_size; }

            private:

            size_t m_size;

            cl_mem m_data;
        };
    }
}