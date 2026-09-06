#include "opencl/impl/Enums.h"

__kernel void propagate(
    __global const float* left_activations, __global float* right_logits, __global float* right_activations,
    __global const float* weights, __global const float* biases,
    const unsigned int batch_size, const unsigned int n_left, const unsigned int n_right,
    const CLActivation activation_function
) {
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    if (col >= n_right || row >= batch_size) return;

    float logit = biases[col];

    for (size_t c = 0; c < n_left; c++) {
        logit += left_activations[row * n_left + c] * weights[c * n_left + col];
    }

    right_logits[row * n_right + col] = logit;

    float activation = logit;

    switch (activation_function) {
        case LEAKY_RELU:
            activation = logit >= 0.0f ? logit : 0.01f * activation;
        break;

        case RELU:
            activation = logit >= 0.0f ? logit : 0.0f;
        break;

        case SIGMOID:
            activation = 1.0f / (1.0f + exp(-logit));
        break;

        case TANH:
            activation = tanh(logit);
        break;

        default: break;
    }

    right_activations[row * n_right + col] = activation;
}