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

    std::vector<size_t> layer_sizes = { 784, 512, 256, 128, 64, 10 };

    mlp::MLP model(context, BATCH_SIZE);
    model.init(layer_sizes);

    mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);

    float test_outputs[30] = {
        0.05f, 0.35f, 0.05f, 0.05f, 0.15f, 0.05f, 0.05f, 0.15f, 0.05f, 0.05f,
        0.05f, 0.05f, 0.15f, 0.05f, 0.35f, 0.05f, 0.05f, 0.05f, 0.05f, 0.15f,
        0.05f, 0.05f, 0.05f, 0.35f, 0.05f, 0.05f, 0.15f, 0.05f, 0.15f, 0.05f
    };

    float test_targets[30] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    mlp::Matrix outputs(3, 10);
    mlp::Matrix targets(3, 10);
    mlp::Matrix results(3, 1);

    context.transfer(test_outputs, outputs);
    context.transfer(test_targets, targets);

    context.cce(outputs, targets, results);
    context.synchronise();

    float host_results[3];

    context.transfer(results, host_results);

    std::cout << host_results[0] << ", " << host_results[1] << ", " << host_results[2] << "\n";

    return 0;
}