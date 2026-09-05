#include "cuda/Context.hpp"
#include "enums/Enums.hpp"

namespace mlp {
    namespace cuda {
        __global__ void cce_kernel(const float* outputs, const float* targets, const size_t classes, const size_t batch_size, float* result) {
            unsigned int sample = blockDim.y * blockIdx.y + threadIdx.y;
            unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (sample >= batch_size || index >= classes || targets[sample * classes + index] != 1.0f) return;

            result[sample] = -logf(outputs[sample * classes + index]);
        }

        void Context::computeLoss(const Matrix_t& outputs, const Matrix_t& targets, Matrix_t& result, const Loss loss) const {
            assert(outputs.size() == targets.size());
            assert(outputs.rows() == targets.rows());
            assert(outputs.columns() == targets.columns());

            cudaError_t err;

            dim3 block(TILE_SIZE, TILE_SIZE);

            dim3 grid(
                block_count(result.rows(), block.x),
                block_count(result.columns(), block.y)
            );

            switch (loss) {
                case Loss::CCE:
                    cce_kernel<<<grid, block>>>(outputs.data(), targets.data(), outputs.columns(), outputs.rows(), result.data());
                break;
            }

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute loss error");
        }
    }
}