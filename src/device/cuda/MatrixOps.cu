// #include <cassert>
// #include <iostream>

// #include <cuda_runtime.h>

// #include "device/CUDAContext.hpp"

// namespace mlp {
//     // used for verifying correctness of the changes I am making
//     __global__ void mat_mul_kernel_old(const float* A, const float* B, float* C, const size_t M, const size_t N, const size_t K) {
//         unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
//         unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

//         if (row >= M || col >= N) return; 

//         float value = 0.0f;

//         for (unsigned int k = 0; k < K; k++) {
//             value += A[row * K + k] * B[k * N + col];
//         }

//         C[row * N + col] = value;
//     }

//     // based on https://siboehm.com/articles/22/CUDA-MMM

//     __global__ void mat_mul_kernel(const float* A, const float* B, float* C, const size_t M, const size_t N, const size_t K) {
//         unsigned int tid = threadIdx.x;

//         unsigned int thread_row = tid / TILE_SIZE;
//         unsigned int thread_col = tid % TILE_SIZE;

//         unsigned int cache_row = blockIdx.y;
//         unsigned int cache_col = blockIdx.x;

//         unsigned int row = cache_row * TILE_SIZE + thread_row;
//         unsigned int col = cache_col * TILE_SIZE + thread_col;

//         __shared__ float As[TILE_SIZE * TILE_SIZE];
//         __shared__ float Bs[TILE_SIZE * TILE_SIZE];

//         A += cache_row * TILE_SIZE * K;
//         B += cache_col * TILE_SIZE;
//         C += cache_row * TILE_SIZE * N + cache_col * TILE_SIZE;

//         float temp = 0.0f;

//         for (unsigned int i = 0; i < K; i += TILE_SIZE) {
//             float newAs = 0.0f;
//             float newBs = 0.0f;

//             if (row < M && i + thread_col < K) newAs = A[thread_row * K + i + thread_col];
//             if (col < N && i + thread_row < K) newBs = B[(thread_row + i) * N + thread_col];

//             As[thread_row * TILE_SIZE + thread_col] = newAs;
//             Bs[thread_row * TILE_SIZE + thread_col] = newBs;

//             __syncthreads();

//             for (unsigned int di = 0; di < TILE_SIZE; di++) {
//                 temp += As[thread_row * TILE_SIZE + di] * Bs[di * TILE_SIZE + thread_col];
//             }

//             __syncthreads();
//         }

//         if (row < M && col < N) {
//             C[thread_row * N + thread_col] = temp;
//         }
//     }

//     __global__ void mat_add_kernel(const float* A, const float* B, float* C, const size_t size) {
//         unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

//         if (index >= size) return;

//         C[index] = A[index] + B[index];
//     }

//     __global__ void mat_add_biases_kernel(const float* A, const float* B, float* C, const size_t M, const size_t N) {
//         unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
//         unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

//         if (row >= M || col >= N) return;

//         C[row * N + col] = A[row * N + col] + B[col];
//     }

//     void CUDAContext::multiply(const CUDAMatrix& matrix_A, const CUDAMatrix& matrix_B, CUDAMatrix& matrix_C) const {
//         assert(matrix_A.columns() == matrix_B.rows());
//         assert(matrix_C.rows() == matrix_A.rows());
//         assert(matrix_C.columns() == matrix_B.columns());

//         cudaError_t err;

//         dim3 block(TILE_SIZE * TILE_SIZE);

//         dim3 grid(
//             block_count(matrix_C.columns(), TILE_SIZE),
//             block_count(matrix_C.rows(), TILE_SIZE)
//         );

//         CUDATaskID task;
//         if (m_profiler) task = m_profiler->startTask("Matrix multiplication      ");

//         mat_mul_kernel<<<grid, block>>>(matrix_A.data(), matrix_B.data(), matrix_C.data(), matrix_A.rows(), matrix_B.columns(), matrix_A.columns());

//         if (m_profiler) m_profiler->endTask(task);

//         err = cudaGetLastError();
//         if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix multiplication error: ");
//     }

//     void CUDAContext::multiplyOld(const CUDAMatrix& matrix_A, const CUDAMatrix& matrix_B, CUDAMatrix& matrix_C) const {
//         assert(matrix_A.columns() == matrix_B.rows());
//         assert(matrix_C.rows() == matrix_A.rows());
//         assert(matrix_C.columns() == matrix_B.columns());

//         cudaError_t err;

//         dim3 block(TILE_SIZE, TILE_SIZE);

//         dim3 grid(
//             block_count(matrix_C.columns(), block.x),
//             block_count(matrix_C.rows(), block.y)
//         );

//         CUDATaskID task;
//         if (m_profiler) task = m_profiler->startTask("Matrix multiplication (old)");

//         mat_mul_kernel_old<<<grid, block>>>(matrix_A.data(), matrix_B.data(), matrix_C.data(), matrix_A.rows(), matrix_B.columns(), matrix_A.columns());

//         if (m_profiler) m_profiler->endTask(task);

//         err = cudaGetLastError();
//         if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix multiplication error: ");
//     }

//     void CUDAContext::add(const CUDAMatrix& matrix_A, const CUDAMatrix& matrix_B, CUDAMatrix& matrix_C) const {
//         assert(matrix_A.size() == matrix_B.size());
//         assert(matrix_B.size() == matrix_C.size());
//         assert(matrix_A.rows() == matrix_B.rows());
//         assert(matrix_B.rows() == matrix_C.rows());
//         assert(matrix_A.columns() == matrix_B.columns());
//         assert(matrix_B.columns() == matrix_C.columns());

//         cudaError_t err;

//         int grid_size = (matrix_A.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

//         CUDATaskID task;
//         if (m_profiler) task = m_profiler->startTask("Matrix addition");

//         mat_add_kernel<<<grid_size, BLOCK_SIZE>>>(matrix_A.data(), matrix_B.data(), matrix_C.data(), matrix_A.size());

//         if (m_profiler) m_profiler->endTask(task);

//         err = cudaGetLastError();
//         if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix addition error: ");
//     }

//     void CUDAContext::addBiases(const CUDAMatrix& layer, const CUDAMatrix& biases, CUDAMatrix& output) const {
//         assert(layer.size() == output.size());
//         assert(layer.rows() == output.rows());
//         assert(layer.columns() == output.columns());
//         assert(biases.rows() == 1);
//         assert(biases.columns() == layer.columns());

//         cudaError_t err;

//         dim3 block(TILE_SIZE, TILE_SIZE);

//         dim3 grid(
//             block_count(output.columns(), block.x),
//             block_count(output.rows(), block.y)
//         );

//         CUDATaskID task;
//         if (m_profiler) task = m_profiler->startTask("Matrix addition (biases)");

//         mat_add_biases_kernel<<<grid, block>>>(layer.data(), biases.data(), output.data(), output.rows(), output.columns());

//         if (m_profiler) m_profiler->endTask(task);

//         err = cudaGetLastError();
//         if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix addition error: ");
//     }
// }