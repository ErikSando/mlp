#include <iostream>

#include <cuda_runtime.h>

#include "device/CUDAContext.hpp"

namespace mlp {
    void CUDAContext::transfer(const CUDAMatrix& src, float* dest) const {
        synchronise();

        CUDATaskID task;
        if (m_profiler) task = m_profiler->startTask("Download");

        cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

        if (m_profiler) m_profiler->endTask(task);
    }

    void CUDAContext::transfer(const float* src, CUDAMatrix& dest) const {
        CUDATaskID task;
        if (m_profiler) task = m_profiler->startTask("Upload");

        cudaError_t err = cudaMemcpy(dest.data(), src, dest.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

        if (m_profiler) m_profiler->endTask(task);
    }

    void CUDAContext::transfer(const CUDAMatrix& src, CUDAMatrix& dest) const {
        CUDATaskID task;
        if (m_profiler) task = m_profiler->startTask("Upload");

        cudaError_t err = cudaMemcpy(dest.data(), src.data(), dest.size() * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

        if (m_profiler) m_profiler->endTask(task);
    }
}