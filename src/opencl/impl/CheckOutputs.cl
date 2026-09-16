__kernel void check_outputs(
    __global const float* outputs, __global const int* labels,
    const unsigned int rows, const unsigned int columns,
    __global int* correct, __global int* classifications
) {
    size_t row = get_global_id(0);

    if (row >= rows) return;

    float max_value = -FLT_MAX;
    int classification = -1;

    for (size_t col = 0; col < columns; col++) {
        if (outputs[row * columns + col] > max_value) {
            max_value = outputs[row * columns + col];
            classification = (int) col;
        }
    }

    classifications[row] = classification;

    if (classification == labels[row]) {
        // atomic_add(correct, 1);
        atomic_inc(correct);
    }
}