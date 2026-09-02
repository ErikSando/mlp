#include "cuda/Context.hpp"

namespace mlp {
    namespace cuda {
        __global__ void check_outputs_kernel(const float* outputs, const int* labels, const size_t rows, const size_t cols, int* correct, int* classifications) {
            unsigned int row = blockIdx.x;

            if (row >= rows) return;

            unsigned int tid = threadIdx.x;
            unsigned int stride = blockDim.x;

            __shared__ float shared_probabilities[BLOCK_SIZE];
            __shared__ int shared_classes[BLOCK_SIZE];

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
            shared_classes[tid] = local_digit;

            __syncthreads();

            unsigned int first_offset = 1;
            while ((first_offset << 1) < min(blockDim.x, (unsigned int) cols)) first_offset <<= 1;

            for (unsigned int offset = first_offset; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    float valueA = shared_probabilities[tid];
                    float valueB = shared_probabilities[tid + offset];

                    int digitA = shared_classes[tid];
                    int digitB = shared_classes[tid + offset];

                    if (valueA >= valueB) {
                        shared_probabilities[tid] = valueA;
                        shared_classes[tid] = digitA;
                        continue;
                    }

                    shared_probabilities[tid] = valueB;
                    shared_classes[tid] = digitB;
                }

                __syncthreads();
            }

            if (tid > 0) return;

            const int classification = shared_classes[0];
            classifications[row] = classification;

            if (classification == labels[row]) {
                atomicAdd(correct, 1);
            }
        }

        __global__ void check_outputs_simple_kernel(const float* outputs, const int* labels, const size_t rows, const size_t cols, int* correct, int* classifications) {
            unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

            if (row >= rows) return;

            float max = -__FLT_MAX__;
            int classification = -1;

            for (size_t col = 0; col < cols; col++) {
                float output = outputs[row * cols + col];
                if (output > max) {
                    max = output;
                    classification = (int) col;
                }
            }

            classifications[row] = classification;

            if (classification == labels[row]) {
                atomicAdd(correct, 1);
            }
        }

        void Context::checkOutputs(const Matrix_t& outputs, const std::vector<int>& labels, Buffer_t& correct, Buffer_t& classifications) const {
            assert(outputs.rows() == labels.size());

            Buffer_t device_labels(labels.size() * sizeof(int));
            transfer(device_labels, (void*) labels.data());

            cudaError_t err;

            // check_outputs_kernel<<<outputs.rows(), BLOCK_SIZE>>>(outputs.data(), (int*) device_labels.data(), outputs.rows(), outputs.columns(), (int*) correct.data(), (int*) classifications.data());

            // using this simple kernel until I can fix the other one

            unsigned int grid_size = block_count(outputs.rows(), BLOCK_SIZE);
            check_outputs_simple_kernel<<<grid_size, BLOCK_SIZE>>>(outputs.data(), (int*) device_labels.data(), outputs.rows(), outputs.columns(), (int*) correct.data(), (int*) classifications.data());

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA check outputs error: ");
        }
    }
}