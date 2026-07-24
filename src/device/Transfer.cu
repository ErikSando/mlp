#include <iostream>

#include <cuda_runtime.h>

#include "device/DeviceContext.hpp"

#define CUDA_ERROR(err, message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message\
                  << cudaGetErrorString(err) << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0)

namespace mlp {
    void DeviceContext::transfer(Matrix& src, float* dest) {
        cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");
    }

    void DeviceContext::transfer(float* src, Matrix& dest) {
        cudaError_t err = cudaMemcpy(dest.data(), src, dest.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");
    }
}