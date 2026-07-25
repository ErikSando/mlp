#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    struct Layer { // purely data for now, maybe I will add a function to pass a previous layer through it
        Layer(const size_t batch_size, const size_t node_count, const size_t previous_count)
        : nodes(batch_size, node_count), weights(previous_count, node_count), biases(1, node_count) {}

        Matrix nodes;
        Matrix weights;
        Matrix biases;
    };
}