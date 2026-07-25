#include <cassert>
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

/*
    thoughts

    i will use a 2D layout, each row is one batch

    first, need to compute the sum of the exponents
    then, divide the exponents by the sun
*/

__global__ void softmax_kernel(const float* input, float* output) { // I will be using input = output but its probably good to seperate the output for versatility

}

namespace mlp {
    void DeviceContext::softmax(const Matrix& input, Matrix& output) {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        dim3 grid(
            block_count(input.columns(), block.x),
            block_count(input.rows(), block.y)
        );

        softmax_kernel<<<grid, block>>>(input.data(), output.data());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA softmax error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }
}