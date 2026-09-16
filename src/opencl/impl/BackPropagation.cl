#include "opencl/impl/Enums.h"

// from https://streamhpc.com/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
inline void atomic_add_f(volatile __global float* source, const float operand) {
    union {
        unsigned int int_val;
        float float_val;
    } new_val;

    union {
        unsigned int int_val;
        float float_val;
    } prev_val;

    do {
        prev_val.float_val = *source;
        new_val.float_val = prev_val.float_val + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int*) source, prev_val.int_val, new_val.int_val) != prev_val.int_val);
}

__kernel void compute_hidden_gradients(
    __global const float* dC_da_gradients,
    __global const float* a_left, __global const float* a_right,
    __global const float* weights,
    const unsigned int n_left, const unsigned int n_right, const unsigned int batch_size,
    __global float* weight_gradients, __global float* bias_gradients, __global float* dC_da_next,
    const CLActivation activation
) {
    size_t left_index = get_global_id(0);
    size_t right_index = get_global_id(1);
    size_t sample = get_global_id(2);

    if (left_index >= n_left || right_index >= n_right || sample >= batch_size) return;

    size_t weight_index = left_index * n_right + right_index;

    float a = a_right[sample * n_right + right_index];
    float aleft = a_left[sample * n_left + left_index];

    float da_dz = 1.0f;

    switch (activation) {
        case LEAKY_RELU:    da_dz = a >= 0.0f ? 1.0f : 0.01f; break;
        case RELU:          da_dz = a >= 0.0f ? 1.0f : 0.0f; break;
        case SIGMOID:       da_dz = a * (1.0f - a); break;
        case TANH:          da_dz = 1.0f - a * a; break;
        default: break;
    }

    float dz_dw = aleft;
    float dC_da = dC_da_gradients[sample * n_right + right_index];

    float dC_dz = dC_da * da_dz;
    float dC_dw = dC_dz * dz_dw;
    float dC_db = dC_dz;

    float dz_da_left = weights[weight_index];
    float dC_da_left = dC_dz * dz_da_left;

    atomic_add_f(&dC_da_next[sample * n_left + left_index], dC_da_left);
    atomic_add_f(&weight_gradients[weight_index], dC_dw / batch_size);

    if (left_index == 0) {
        atomic_add_f(&bias_gradients[right_index], dC_db / batch_size);
    }
}

__kernel void compute_output_gradients(
    __global const float* hidden_activations, __global const float* output_activations,
    __global const float* weights,
    const unsigned int n_hidden, const unsigned int n_output, const unsigned int batch_size,
    __global const int* labels, __global float* weight_gradients, __global float* bias_gradients, __global float* dC_da_hidden_list,
    const CLActivation activation, const CLLoss loss
) {
    size_t hidden_index = get_global_id(0);
    size_t output_index = get_global_id(1);
    size_t sample = get_global_id(2);

    if (hidden_index >= n_hidden || output_index >= n_output || sample >= batch_size) return;

    size_t weight_index = hidden_index * n_output + output_index;

    float a_output = output_activations[sample * n_output + output_index];
    float a_hidden = hidden_activations[sample * n_hidden + hidden_index];

    // currently assuming Softmax + CCE

    float y = 0.0f;
    if (output_index == labels[sample]) y = 1.0f;

    float dC_dz_output = a_output - y;
    float dC_dw_output = a_hidden * dC_dz_output;
    float dC_db_output = dC_dz_output;

    float dz_output_da_hidden = weights[weight_index];

    float dC_da_hidden = dC_dz_output * dz_output_da_hidden;

    atomic_add_f(&dC_da_hidden_list[sample * n_hidden + hidden_index], dC_da_hidden);
    atomic_add_f(&weight_gradients[weight_index], dC_dw_output / batch_size);

    if (hidden_index == 0) {
        atomic_add_f(&bias_gradients[output_index], dC_db_output / batch_size);
    }
}

__kernel void optimise_layer(
    __global float* weights, __global float* biases,
    __global const float* weight_gradients, __global const float* bias_gradients,
    const unsigned int n_weights, const unsigned int n_biases,
    const float learning_rate
) {
    size_t index = get_global_id(0);

    if (index >= n_weights) return;

    weights[index] -= weight_gradients[index] * learning_rate;

    if (index < n_biases) biases[index] -= bias_gradients[index] * learning_rate;
}
