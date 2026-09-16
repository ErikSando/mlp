#include "opencl/impl/Config.h"

__kernel void softmax(
    __global float* logits,
    __global float* activations,
    const unsigned int rows,
    const unsigned int columns
) {
    // launch with global size of the number of rows
    // local 256

    size_t row = get_group_id(0);

    if (row >= rows) return;

    size_t tid = get_local_id(0);
    size_t stride = get_local_size(0); // should equal SOFTMAX_WORK_GROUP_SIZE

    __local float shared[SOFTMAX_WORK_GROUP_SIZE]; // shared between threads in the same row (same get_global_id(0))

    float local_max = logits[row * columns + tid];

    for (size_t col = tid + stride; col < columns; col += stride) {
        if (logits[row * columns + col] > local_max) {
            local_max = logits[row * columns + col];
        }
    }

    shared[tid] = local_max;

    work_group_barrier(CLK_GLOBAL_MEM_FENCE);

    size_t first_offset = 1;
    while ((first_offset << 1) < min(stride, (size_t) columns)) first_offset <<= 1;

    for (size_t offset = first_offset; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (shared[tid + offset] > shared[tid]) {
                shared[tid] = shared[tid + offset];
            }
        }

        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }

    const float max_value = shared[0];

    float partial_sum = 0.0f;

    for (size_t col = tid; col < columns; col += stride) {
        partial_sum += exp(logits[row * columns + col] - max_value);
    }

    shared[tid] = partial_sum;

    work_group_barrier(CLK_GLOBAL_MEM_FENCE);

    for (size_t offset = first_offset; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }

        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }

    const float sum = shared[0];

    for (size_t col = tid; col < columns; col += stride) {
        activations[row * columns + col] = exp(logits[row * columns + col] - max_value) / sum;
    }
}