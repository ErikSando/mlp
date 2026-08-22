#include <iostream>

#include <cuda_runtime.h>

#include "cuda/memory/Buffer.hpp"

namespace mlp {
    namespace cuda {
        Buffer::Buffer(const size_t size) : m_size(size) {
            cudaError_t err = cudaMalloc((void**) &m_data, m_size);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        }

        Buffer::~Buffer() {
            cudaFree(m_data);
        }

        void Buffer::zero() {
            cudaError_t err = cudaMemset(m_data, 0, m_size);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}