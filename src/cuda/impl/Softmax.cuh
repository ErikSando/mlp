#pragma once

namespace mlp {
    __global__ void softmax_kernel(const float* logits, float* activations, const size_t rows, const size_t cols);
}