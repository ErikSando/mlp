#include <iostream>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"

#include "profiling/CUDAProfiler.hpp"
#include "profiling/Profiler.hpp"

int main() {
    constexpr size_t BATCH_SIZE = 1024;

    mlp::CUDAProfiler cuda_profiler;
    mlp::Profiler profiler;

    mlp::Dataset training("res/mnist/mnist_train.csv");
    mlp::DeviceContext context(&cuda_profiler);

    // std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };
    std::vector<size_t> layer_sizes = { 784, 2048, 2048, 1024, 512, 256, 10 };

    mlp::MLP mlp(context, BATCH_SIZE);
    mlp.init(layer_sizes);

    mlp::Batch batch(BATCH_SIZE, 784);

    // I'm using the profiler like a benchmarker here? I don't really know the meanings
    // I think benchmarking is measuring the time taken to complete a task
    // And profiling is measuring the share of computation time of seperate tasks

    // profiler.startTask("Batch parsing");
    training.parseBatch(batch);
    // profiler.endTask();

    // profiler.startTask("Forward pass");
    mlp.forwardPass(batch);
    // profiler.endTask();

    // float* outputs = new float[BATCH_SIZE * 10];

    // mlp.copyOutputs(outputs);

    // for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
    //     std::cout << "Batch " << batch << ":\n";
    //     std::cout << outputs[0 + batch * 10];

    //     for (size_t i = 1; i < 10; i++) {
    //         std::cout << ", " << outputs[i + batch * 10];
    //     }

    //     std::cout << "\n\n";
    // }

    // delete[] outputs;

    cuda_profiler.disable();

    profiler.startTask("Batch parsing");
    for (int i = 0; i < 100; i++) {
        training.parseBatch(batch);
    }
    profiler.endTask();

    profiler.startTask("Forward passes");
    for (int i = 0; i < 100; i++) {
        mlp.forwardPass(batch);
    }
    profiler.endTask();

    profiler.print();

    cuda_profiler.print();

    return 0;
}