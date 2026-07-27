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

    constexpr size_t BATCH_SIZE = 1024;

    std::vector<size_t> layer_sizes = { 784, 512, 256, 128, 64, 10 };

    mlp::MLP model(context, BATCH_SIZE);
    model.init(layer_sizes);

    // cuda_profiler.disable();

    mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);

    // profiler is not designed to measure tasks carried out in alternation, so seperating the benchmarking

    profiler.startTask("Batch parse       ");
    for (int i = 0; i < 100; i++) {
        dataset.parseBatch(batch);
    }
    profiler.endTask();

    std::vector<float> old_results(BATCH_SIZE * layer_sizes.back());
    std::vector<float> new_results(BATCH_SIZE * layer_sizes.back());

    // profiler.startTask("Forward pass (old)");
    // for (int i = 0; i < 100; i++) {
        model.forwardPassOld(batch);
    // }
    // profiler.endTask();

    model.copyOutputs(old_results.data());

    // profiler.startTask("Forward pass      ");
    // for (int i = 0; i < 100; i++) {
        model.forwardPass(batch);
    // }
    // profiler.endTask();

    model.copyOutputs(new_results.data());

    profiler.print();
    cuda_profiler.print();

    for (int i = 0; i < old_results.size(); i++) {
        if (old_results[i] != new_results[i]) {
            break;
            std::cout << "Incorrect result\n";
        }
    }

    return 0;
}