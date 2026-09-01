#include <iostream>

#include <cuda_runtime.h>

#include "cuda/Context.hpp"

namespace mlp {
    namespace cuda {
        void Context::transfer(float* dest, const Matrix_t& src) const {
            synchronise();

            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Download");

            cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

            if (m_profiler) m_profiler->endTask(task);
        }

        void Context::transfer(Matrix_t& dest, const float* src) const {
            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Upload");

            cudaError_t err = cudaMemcpy(dest.data(), src, dest.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

            if (m_profiler) m_profiler->endTask(task);
        }

        void Context::transfer(Matrix_t& dest, const Matrix_t& src) const {
            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Upload");

            cudaError_t err = cudaMemcpy(dest.data(), src.data(), dest.size(), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

            if (m_profiler) m_profiler->endTask(task);
        }

        void Context::transfer(void* dest, const Buffer_t& src) const {
            // synchronise();

            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Download");

            cudaError_t err = cudaMemcpy(dest, src.data(), src.size(), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

            if (m_profiler) m_profiler->endTask(task);
        }

        void Context::transfer(Buffer_t& dest, const void* src) const {
            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Upload");

            cudaError_t err = cudaMemcpy(dest.data(), src, dest.size(), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

            if (m_profiler) m_profiler->endTask(task);
        }

        void Context::transfer(Buffer_t& dest, const Buffer_t& src) const {
            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Upload");

            cudaError_t err = cudaMemcpy(dest.data(), src.data(), dest.size(), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

            if (m_profiler) m_profiler->endTask(task);
        }
    }
}