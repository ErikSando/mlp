#include "opencl/Context.hpp"
#include "opencl/CLstuff.hpp"

namespace mlp {
    namespace opencl {
        void Context::transfer(Matrix_t& dest, const float* src) const {
            // if (m_profiler) m_profiler->startTask("Upload");

            cl_int err = clEnqueueWriteBuffer(command_queue, dest.data(), CL_FALSE, 0, dest.size() * sizeof(float), (void*) src, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to transfer from host to device");

            // if (m_profiler) m_profiler->endTask("Upload");
        }

        void Context::transfer(float* dest, const Matrix_t& src) const {
            // if (m_profiler) m_profiler->startTask("Download");

            cl_int err = clEnqueueReadBuffer(command_queue, src.data(), CL_FALSE, 0, src.size() * sizeof(float), (void*) dest, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to transfer from device to host");

            // if (m_profiler) m_profiler->endTask("Download");
        }

        void Context::transfer(Matrix_t& dest, const Matrix_t& src) const {
            // if (m_profiler) m_profiler->startTask("Copy");

            assert(src.size() == dest.size());

            cl_int err = clEnqueueCopyBuffer(command_queue, src.data(), dest.data(), 0, 0, src.size() * sizeof(float), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to copy device memory into another location");

            // if (m_profiler) m_profiler->endTask("Copy");
        }

        void Context::transfer(Buffer_t& dest, const void* src) const {
            // if (m_profiler) m_profiler->startTask("Upload");

            cl_int err = clEnqueueWriteBuffer(command_queue, dest.data(), CL_FALSE, 0, dest.size(), src, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to transfer from host to device");

            // if (m_profiler) m_profiler->endTask("Upload");
        }

        void Context::transfer(void* dest, const Buffer_t& src) const {
            // if (m_profiler) m_profiler->startTask("Download");

            cl_int err = clEnqueueReadBuffer(command_queue, src.data(), CL_FALSE, 0, src.size(), dest, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to transfer from device to host");

            // if (m_profiler) m_profiler->endTask("Download");
        }

        void Context::transfer(Buffer_t& dest, const Buffer_t& src) const {
            // if (m_profiler) m_profiler->startTask("Copy");

            assert(src.size() == dest.size());

            cl_int err = clEnqueueCopyBuffer(command_queue, src.data(), dest.data(), 0, 0, src.size(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to copy device memory into another location");

            // if (m_profiler) m_profiler->endTask("Copy");
        }
    }
}