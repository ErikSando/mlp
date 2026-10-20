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

__kernel void cce(
    __global const float* outputs, __global const float* targets,
    const unsigned int n_classes, const unsigned int batch_size,
    __global float* result
) {
    size_t sample = get_global_id(0);
    size_t index = get_global_id(1);

    if (sample >= batch_size || index >= n_classes || targets[sample * n_classes + index] != 1.0f) return;

    result[sample] = -log(outputs[sample * n_classes + index]);
}

__kernel void mse(
    __global const float* outputs, __global const float* targets,
    const unsigned int n_classes, const unsigned int batch_size,
    __global float* result
) {
    size_t sample = get_global_id(0);
    size_t index = get_global_id(1);

    if (sample >= batch_size || index >= n_classes) return;

    float error = outputs[sample * n_classes + index] - targets[sample * n_classes + index];

    atomic_add_f(&result[sample], error * error);

    work_group_barrier(CLK_GLOBAL_MEM_FENCE);

    if (index == 0) {
        result[sample] = result[sample] / n_classes;
    }
}