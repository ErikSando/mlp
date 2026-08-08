#pragma once

#include <vector>

namespace mlp {
    constexpr int NO_LABEL = -1;

    struct Sample {
        Sample(const size_t size) : data(size) {}

        int label = NO_LABEL;
        std::vector<float> data;
    };
}