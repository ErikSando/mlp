#pragma once

#include <cstddef>

namespace mlp {
    void softmax_op(const float* inputs, float* outputs, const size_t rows, const size_t cols);
}