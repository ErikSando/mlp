#pragma once

#include <vector>

namespace mlp {
    struct Sample {
        Sample(const size_t size) : data(size) {}

        int label;
        std::vector<float> data;
    };
}