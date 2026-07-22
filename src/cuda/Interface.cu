#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Interface.hpp"

__global__ void init_rng(curandState* states, unsigned long seed, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        curand_init(seed, index, 0, &states[index]);
    }
}

/*
    Calculate C = AB
    A: matrix of order M x K
    B: matrix of order K x N
    C: matrix of order M x N
*/
__global__ void mat_mul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) return; 

    float value = 0.0f;

    for (int k = 0; k < K; k++) {
        value += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = value;
}

__global__ void randomize_kernel(float* M, curandState* states, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        int value = static_cast<int>(curand(&states[index]) % 21) - 10;

        M[index] = static_cast<float>(value);
    }
}

namespace mlp {
    void Interface::multiply(Matrix& mA, Matrix& mB, Matrix& mC) {
        assert(mA.columns() == mB.rows());
        assert(mC.rows() == mA.rows());
        assert(mC.columns() == mB.columns());

        dim3 block(16, 16);

        dim3 grid(
            (mB.columns() + block.x - 1) / block.x,
            (mA.rows() + block.y - 1) / block.y
        );

        mat_mul_kernel<<<grid, block>>>(mA.data(), mB.data(), mC.data(), mA.rows(), mB.columns(), mA.columns());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "Matrix multiplication error: " << cudaGetErrorString(err) << '\n';

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "CUDA device synchronize error: " << cudaGetErrorString(err) << '\n';
    }

    void Interface::randomise(Matrix& m) {
        constexpr int blockSize = 256;
        int gridSize = (m.size() + blockSize - 1) / blockSize;

        static curandState* states = nullptr;
        static int allocated = 0;

        if (allocated < m.size()) {
            if (states) cudaFree(states);

            cudaMalloc(&states, m.size() * sizeof(curandState));

            init_rng<<<gridSize, blockSize>>>(states, time(nullptr), m.size());

            allocated = m.size();
        }

        randomize_kernel<<<gridSize, blockSize>>>(m.data(), states, m.size());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "Matrix fill error: " << cudaGetErrorString(err) << '\n';
    
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) std::cerr << "CUDA device synchronize error: " << cudaGetErrorString(err) << '\n';
    }

    void Interface::transfer(Matrix& src, float* dest) {
        cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << '\n';
    }
}