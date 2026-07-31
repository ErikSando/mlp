#include "device/DeviceContext.hpp"

namespace mlp {
    __global__ void cce_kernel(const float* output, const float* target, const size_t classes, const size_t batch_size, float* result) {
        unsigned int batch = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (batch >= batch_size || index >= classes || target[batch * classes + index] != 1.0f) return;

        result[batch] = -logf(output[batch * classes + index]);
    }

    void DeviceContext::cce(const Matrix& output, const Matrix& target, Matrix& result) const {
        assert(output.size() == target.size());
        assert(output.rows() == target.rows());
        assert(output.columns() == target.columns());

        cudaError_t err;

        dim3 block(TILE_SIZE, TILE_SIZE);

        dim3 grid(
            block_count(result.rows(), block.x),
            block_count(result.columns(), block.y)
        );

        cce_kernel<<<grid, block>>>(output.data(), target.data(), output.columns(), output.rows(), result.data());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA layer propagation error: ");
    }

    void DeviceContext::computeLoss(const Matrix& output, const Matrix& target, Matrix& result, const Loss loss) const {
        assert(output.size() == target.size());
        assert(output.rows() == target.rows());
        assert(output.columns() == target.columns());

        cudaError_t err;

        dim3 block(TILE_SIZE, TILE_SIZE);

        dim3 grid(
            block_count(result.rows(), block.x),
            block_count(result.columns(), block.y)
        );

        switch (loss) {
            case Loss::MSE:

            break;

            case Loss::CCE:
                cce_kernel<<<grid, block>>>(output.data(), target.data(), output.columns(), output.rows(), result.data());
            break;

            default: break;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA layer propagation error: ");
    }
}