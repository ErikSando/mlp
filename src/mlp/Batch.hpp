#pragma once

#include <vector>

// not really sure how to design array sizing and stuff, so for now im just going to use vectors and make everything flexible
// will optimise later maybe

// purpose:
// store all the data of a batch of input layers (batch size, labels, and input node values)

namespace mlp {
    // stores input data on the CPU
    struct Batch {
        /*
            batch_size: number of input layers
            input_count: number of input nodes per input layer
        */
        Batch(size_t batch_size, size_t input_count) : size(batch_size), labels(batch_size), data(batch_size * input_count) {}

        size_t size;
        std::vector<int> labels;
        std::vector<float> data;
    };
}