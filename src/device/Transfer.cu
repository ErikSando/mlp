#include <iostream>

#include <cuda_runtime.h>

#include "device/DeviceContext.hpp"

namespace mlp {
    void DeviceContext::transfer(const Matrix& src, float* dest) const {
        CUDATaskID task;
        if (m_profiler) task = m_profiler->startTask("Download");

        cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

        if (m_profiler) m_profiler->endTask(task);
    }

    void DeviceContext::transfer(const float* src, Matrix& dest) const {
        CUDATaskID task;
        if (m_profiler) task = m_profiler->startTask("Upload");

        cudaError_t err = cudaMemcpy(dest.data(), src, dest.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");

        if (m_profiler) m_profiler->endTask(task);
    }
}