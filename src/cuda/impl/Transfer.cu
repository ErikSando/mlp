#include <iostream>

#include <cuda_runtime.h>

#include "cuda/Context.hpp"

namespace mlp {
    namespace cuda {
        void Context::transfer(float* dest, const Matrix& src) const {
            // synchronise();

            if (m_profiler) m_profiler->startTask("Download");

            cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error");

            if (m_profiler) m_profiler->endTask("Download");
        }

        void Context::transfer(Matrix& dest, const float* src) const {
            if (m_profiler) m_profiler->startTask("Upload");

            cudaError_t err = cudaMemcpy(dest.data(), src, dest.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error");

            if (m_profiler) m_profiler->endTask("Upload");
        }

        void Context::transfer(Matrix& dest, const Matrix& src) const {
            if (m_profiler) m_profiler->startTask("Copy");

            assert(src.size() == dest.size());

            cudaError_t err = cudaMemcpy(dest.data(), src.data(), dest.size(), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error");

            if (m_profiler) m_profiler->endTask("Copy");
        }

        void Context::transfer(void* dest, const Buffer& src) const {
            // synchronise();

            if (m_profiler) m_profiler->startTask("Download");

            cudaError_t err = cudaMemcpy(dest, src.data(), src.size(), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error");

            if (m_profiler) m_profiler->endTask("Download");
        }

        void Context::transfer(Buffer& dest, const void* src) const {
            if (m_profiler) m_profiler->startTask("Upload");

            cudaError_t err = cudaMemcpy(dest.data(), src, dest.size(), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error");

            if (m_profiler) m_profiler->endTask("Upload");
        }

        void Context::transfer(Buffer& dest, const Buffer& src) const {
            if (m_profiler) m_profiler->startTask("Copy");

            assert(src.size() == dest.size());

            cudaError_t err = cudaMemcpy(dest.data(), src.data(), dest.size(), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error");

            if (m_profiler) m_profiler->endTask("Copy");
        }
    }
}