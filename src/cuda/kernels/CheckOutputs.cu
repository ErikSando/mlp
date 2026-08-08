#include "cuda/Context.hpp"

namespace mlp {
    namespace cuda {
        __global__ void check_outputs_kernel(const float* outputs, const int* labels, const size_t rows, const size_t cols, const size_t n_samples, float* correct, float* classifications) {
            unsigned int row = blockIdx.x;

            if (row >= rows) return;

            unsigned int tid = threadIdx.x;
            unsigned int stride = blockDim.x;

            __shared__ float shared_probabilities[BLOCK_SIZE];
            __shared__ int shared_digits[BLOCK_SIZE];

            float local_max = -__FLT_MAX__;
            int local_digit = -1;

            for (unsigned int col = tid; col < cols; col += stride) {
                float value = outputs[row * cols + col];

                if (value > local_max) {
                    local_digit = col;
                    local_max = value;
                }
            }

            shared_probabilities[tid] = local_max;
            shared_digits[tid] = local_digit;

            __syncthreads();

            unsigned int first_offset = 1;
            while ((first_offset << 1) < min(blockDim.x, (unsigned int) cols)) first_offset <<= 1;

            for (unsigned int offset = first_offset; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    float valueA = shared_probabilities[tid];
                    float valueB = shared_probabilities[tid + offset];

                    int digitA = shared_digits[tid];
                    int digitB = shared_digits[tid + offset];

                    if (valueA >= valueB) {
                        shared_probabilities[tid] = valueA;
                        shared_digits[tid] = digitA;
                        continue;
                    }

                    shared_probabilities[tid] = valueB;
                    shared_digits[tid] = digitB;
                }

                __syncthreads();
            }

            if (tid > 0) return;

            const int classification = shared_digits[0];
            classifications[row] = (float) classification;

            if (classification == labels[row]) {
                atomicAdd(correct, 1.0f);
            }
        }

        void Context::checkOutputs(const Matrix& outputs, const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications) const {
            assert(outputs.rows() == labels.size());
            assert(n_samples <= labels.size());

            cudaError_t err;

            check_outputs_kernel<<<outputs.rows(), BLOCK_SIZE>>>(outputs.data(), labels.data(), outputs.rows(), outputs.columns(), n_samples, correct.data(), classifications.data());

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA check outputs error: ");
        }
    }
}