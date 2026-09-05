#pragma once

#include <cstring>

#include "CL/cl.h"

namespace mlp {
    namespace opencl {
        class Kernel {
            public:

            Kernel(const std::string& source_path, const std::string& name);
            ~Kernel();

            template<typename T>
            cl_int setArgument(cl_uint index, const T& value) {
                return clSetKernelArg(m_clKernel, index, sizeof(T), &value);
            }

            template<typename T>
            cl_int setLocalArrayArgument(cl_uint index, const size_t size) {
                return clSetKernelArg(m_clKernel, index, size * sizeof(T), nullptr);
            }

            const cl_kernel& getCLKernel() const { return m_clKernel; }

            private:

            cl_program m_clProgram = nullptr;
            cl_kernel m_clKernel = nullptr;
        };
    }
}