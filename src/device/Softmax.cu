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

__global__ void softmax_kernel(const float* input, float* output, const size_t rows, const size_t cols) { // I will be using input = output but its probably good to seperate the output for versatility
    unsigned int row = blockIdx.x; // i dont think the choice between unsigned int or size_t matters

    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;

    __shared__ float shared[mlp::BLOCK_SIZE]; // size of shared[] needs to match blockDim.x, which is equal to mlp::BLOCK_SIZE

    float local_max = -__FLT_MAX__;

    // The articles about softmax were very difficult to understand and I was impatient so I used ChatGPT to explain how this stuff works
    // But I haven't used warp-level reduction yet, so I could perhaps improve it further

    /*
        Each block is assigned a row
        Each thread is assigned different columns, the columns assigned to each thread are tid, tid + stride, tid + stride * 2, ...
        The loop below computes the maximum value out of these assigned columns, and stores it in shared[tid] (assuming the thread is assigned more than 1 column)

        For how I'm using it, we skip tid >= 10 as we have 10 classes (C = 10)
        The stride is far larger than the number of columns so each thread is assigned to one column and will store the value in that column in shared[tid]
        Each thread from tid = 0 ... 9 will execute one iteration of the loop, so the first 10 values in shared[] represent the row assigned to this block
        Every value after that ends up with negative __FLT_MAX__
        For C < stride, no reduction occurs yet
    */

    for (unsigned int col = tid; col < cols; col += stride) {
        float value = input[row * cols + col];
        local_max = fmaxf(local_max, value);
    }

    shared[tid] = local_max;

    __syncthreads();

    /*
        Here we want to perform reduction to find the maximum value out of the values stored in shared[]

        Because max is associative, e.g. max(a, max(b, c)) == max(max(a, b), c), we can perform max on any two values in shared[] at a time
        The process can be visualised like an upside down binary tree, using max to converge two nodes in a layer into one node in the next layer
        Eventually we end up with the max value in shared[] at shared[0], which is the maximum value in the row

        The loop below creates a variable offset starting at half the thread count, halving it each iteration
        For any thread with tid < offset, we compute the max between the value it assigned to shared[], which is shared[tid], and shared[tid + offset]
        We check tid < offset because we only need to check half of all the values (because we compare two values at a time)

        Assuming we don't skip the first values of offset (blockDim.x / 2, blockDim.x / 4, etc):

        In the case of C = 10, the first iteration looks like

        shared[0] = max(shared[0], shared[128]) = max(shared[0], -__FLT_MAX) = shared[0]
        shared[1] = max(shared[1], shared[129]) = max(shared[1], -__FLT_MAX) = shared[1]
        ...
        shared[9] = max(shared[9], shared[137]) = max(shared[9], -__FLT_MAX) = shared[9]

        Until offset is small enough, these comparisions won't be changing any values in shared[]

        As offset is halved each time, we eventually end up comparing two values in shared[] with 0 <= tid <= 9
        This would be when offset <= 8, and because the thread count is a power of 2, we will end up at 8 eventually
        We can skip to this and prevent wasting time with the redundant larger values of offset

        The first offset will be the largest power of 2 smaller than the size of shared[] (so it will be at least at least half)
        So the largest shared[] index will be <= 2 * offset - 1
        We use max(shared[tid], shared[tid + offset]), with tid < offset, meaning up to offset - 1 + offset = 2 * offset - 1, so all indices of shared[] are included

        I doubt the time save will be anything remarkable, though

        The process then becomes:

        shared[0] = max(shared[0], shared[8])
        shared[1] = max(shared[1], shared[9])
        ... the rest remain as shared[tid]

        shared[8] and shared[9] are now either eliminated or placed in shared[0] / shared[1], so we don't need to compare with them again

        next:

        shared[0] = max(shared[0], shared[4])
        shared[1] = max(shared[1], shared[5])
        shared[2] = max(shared[2], shared[6])
        shared[3] = max(shared[3], shared[7])
        ... the rest are redundant,
            but shared[4] and shared[5] will be compared with shared[8] and shared[9], respectively
            I haven't thought of a cheap way to eliminate comparisons with values already compared, so I'll just leave them (everything is parallelised so I don't think it's even beneficial)

        Now every value has been compared and reduced into shared[0..3]

        next:

        shared[0] = max(shared[0], shared[2])
        shared[1] = max(shared[1], shared[3])
        ...

        and finally:

        shared[0] = max(shared[0], shared[1])

        and the largest value in shared[0...9] is stored in shared[0]
    */

    // I'm going up from 1 instead of going down from blockDim.x / 2 because the offset needs to be a power of 2
    // I don't think blockDim.x strictly needs to be a power of 2 so I won't assume it is
    // WARNING: I realised if cols > blockDim.x, then blockDim.x has to be an exponent of 2
    // If it isn't, then later tid + offset will be >= blockDim.x and shared[tid + offset] will be out of bounds
    unsigned int first_offset = 1;
    while ((first_offset << 1) < min(blockDim.x, (unsigned int) cols)) first_offset <<= 1; // multiplication by a power of 2 probably compiles to a left bit shift anyway, but this is cool

    for (unsigned int offset = first_offset; offset > 0; offset >>= 1) {
        // if tid + offset >= cols, then the comparison is redundant
        // but adding a condition to check this will probably slow down the kernel
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }

        __syncthreads();
    }

    const float max_value = shared[0];

    /*
        Now we compute the sum of exponents

        I forgot to mention we are subtracting each exponent by the maximum found earlier to make the softmax function numerically stable by keeping the exponents small

        In our case:

        The loop below will be executed on tid = 0...9
        And the exponent of the value in column tid will be assigned to shared[tid]

        (keep in mind, if columns > blockDim.x, then the values in shared[tid] will be a sum of exponents of multiple columns)

        shared[0...9] are the exponents of each column

        With larger values of C there will be some additional reduction here, but not yet with C = 10
    */

    float partial_sum = 0.0f;

    for (unsigned int col = tid; col < cols; col += stride) {
        partial_sum += expf(input[row * cols + col] - max_value);
    }

    shared[tid] = partial_sum;

    __syncthreads();

    /*
        Here we use the same concept used earlier but max is replaced with sum (both are associative)

        Two values in shared[] are summed and stored in the location closer to 0
        Eventually shared[0] will equal the sum of the exponent of all relevent values
    */

    for (unsigned int offset = first_offset; offset > 0; offset >>= 1) {
        if (tid < offset) { // values in shared[>=10] will be 0, so we don't need to skip them using tid + offset < cols (this might be worse because it adds an instruction)
            shared[tid] += shared[tid + offset];
        }

        __syncthreads();
    }

    const float sum = shared[0];

    // And then we can assign exp(x - max) / sum(exp(x - max)) to the values in the output matrix

    for (unsigned int col = tid; col < cols; col += stride) {
        output[row * cols + col] = expf(input[row * cols + col] - max_value) / sum;
    }
}

namespace mlp {
    void DeviceContext::softmax(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());
        assert(input.columns() < BLOCK_SIZE);

        cudaError_t err;

        // BLOCK_SIZE is definitely far larger than needed for a digit classification model, but I don't think it matters
        softmax_kernel<<<input.rows(), BLOCK_SIZE>>>(input.data(), output.data(), input.rows(), input.columns());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA softmax error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }
}