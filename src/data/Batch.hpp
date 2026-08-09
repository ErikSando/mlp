#pragma once

#include <vector>

// #include "data/Sample.hpp"

namespace mlp {
    // stores input data in host memory
    struct Batch {
        /*
            batch_size: number of input layers
            input_count: number of input nodes per input layer
        */
        Batch(size_t batch_size, size_t input_count) : size(batch_size), labels(batch_size), data(batch_size * input_count) {}

        size_t size;
        std::vector<int> labels;
        std::vector<float> data;

        // std::vector<Sample> samples;
    };
}