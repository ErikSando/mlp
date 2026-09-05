#include "opencl/Context.hpp"

namespace mlp {
    namespace opencl {
        void Context::transfer(Matrix_t& dest, const float* src) const {
            // if (m_profiler) m_profiler->startTask("Upload");

            cl_int err = clEnqueueWriteBuffer(m_commandQueue, dest.data(), CL_FALSE, 0, dest.size() * sizeof(float), (void*) src, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to transfer from host to device");

            // if (m_profiler) m_profiler->endTask("Upload");
        }

        void Context::transfer(float* dest, const Matrix_t& src) const {
            // if (m_profiler) m_profiler->startTask("Download");

            cl_int err = clEnqueueReadBuffer(m_commandQueue, src.data(), CL_FALSE, 0, src.size() * sizeof(float), (void*) dest, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to transfer from device to host");

            // if (m_profiler) m_profiler->endTask("Download");
        }

        void Context::transfer(Matrix_t& dest, const Matrix_t& src) const {
            // if (m_profiler) m_profiler->startTask("Copy");

            // if (m_profiler) m_profiler->endTask("Copy");
        }

        void Context::transfer(Buffer_t& dest, const void* src) const {
            // if (m_profiler) m_profiler->startTask("Upload");

            // if (m_profiler) m_profiler->endTask("Upload");
        }

        void Context::transfer(void* dest, const Buffer_t& src) const {
            // if (m_profiler) m_profiler->startTask("Download");

            // if (m_profiler) m_profiler->endTask("Download");
        }

        void Context::transfer(Buffer_t& dest, const Buffer_t& src) const {
            // if (m_profiler) m_profiler->startTask("Copy");

            // if (m_profiler) m_profiler->endTask("Copy");
        }
    }
}