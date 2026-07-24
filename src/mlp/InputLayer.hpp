#pragma once

#include <vector>

// not really sure how to design array sizing and stuff, so for now im just going to use vectors and make everything flexible
// will optimise later maybe

namespace mlp {
    // stores input data on the CPU
    struct InputLayer {
        InputLayer(size_t count) : data(count) {}
        int label;
        std::vector<float> data;
    };
}