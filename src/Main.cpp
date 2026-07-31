#include <iostream>

#include <random>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"
#include "profiling/Profiler.hpp"

int main() {
    mlp::Dataset dataset("res/mnist/mnist_train.csv");

    mlp::CUDAProfiler cuda_profiler;
    mlp::Profiler profiler;

    mlp::DeviceContext context(&cuda_profiler);

    constexpr size_t BATCH_SIZE = 32;

    std::vector<size_t> layer_sizes = { 784, 2048, 2048, 1024, 512, 256, 128, 64, 10 };

    mlp::MLP model(context, BATCH_SIZE);
    model.init(layer_sizes);

    mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);
    dataset.parseBatch(batch);

    profiler.startTask("Forward passes");

    for (int i = 0; i < 500; i++) {
        model.forwardPass(batch);
    }

    float* host_outputs = new float[BATCH_SIZE * layer_sizes.back()];
    model.copyOutputs(host_outputs);

    profiler.endTask();
    profiler.print();

    for (int b = 0; b < BATCH_SIZE; b++) {
        std::cout << "Batch " << b + 1 << ":\n";
        for (int i = 0; i < layer_sizes.back(); i++) {
            std::cout << "  " << host_outputs[b * layer_sizes.back() + i];
        }
        std::cout << "\n\n";
    }

    delete[] host_outputs;

    model.backwardPass();

    return 0;
}