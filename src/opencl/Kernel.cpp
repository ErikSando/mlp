#include <fstream>
#include <iostream>
#include <sstream>

#include "opencl/CLstuff.hpp"
#include "opencl/Kernel.hpp"

namespace mlp {
    namespace opencl {
        std::string read_file(const std::string& file_path) {
            std::ifstream file(file_path);

            if (!file.is_open()) {
                std::cerr << "Cannot open file: " << file_path << std::endl;
                return "";
            }

            std::stringstream contents;
            contents << file.rdbuf();

            return contents.str();
        }

        Kernel::Kernel(const std::string& source_path, const std::string& name) {
            std::string kernel_src = read_file(source_path);

            if (!kernel_src.size()) {
                ERROR("Failed to read kernel source");
            }

            cl_int err;

            const char* src = kernel_src.c_str();

            m_clProgram = clCreateProgramWithSource(clcontext, 1, (const char**) &src, nullptr, &err);

            if (!m_clProgram) {
                CL_ERROR("Failed to create compute program", err);
            }

            err = clBuildProgram(m_clProgram, 0, nullptr, nullptr, nullptr, nullptr);

            if (err != CL_SUCCESS) {
                size_t length;
                clGetProgramBuildInfo(m_clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
                char buffer[length];
                clGetProgramBuildInfo(m_clProgram, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, nullptr);
                CL_ERROR("Failed to build program executible", err);
                std::cout << buffer << "\n";
                return;
            }

            m_clKernel = clCreateKernel(m_clProgram, name.c_str(), &err);

            if (err != CL_SUCCESS) {
                CL_ERROR("Failed to create compute kernel", err);
            }

            if (!m_clKernel) {
                ERROR("Failed to create compute kernel");
            }
        }

        Kernel::~Kernel() {
            if (m_clKernel) clReleaseKernel(m_clKernel);
            if (m_clProgram) clReleaseProgram(m_clProgram);
        }
    }
}