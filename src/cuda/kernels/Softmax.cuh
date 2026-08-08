#pragma once

namespace mlp {
    namespace cuda {
        __global__ void softmax_kernel(const float* inputs, float* outputs, const size_t rows, const size_t cols);
    }
}