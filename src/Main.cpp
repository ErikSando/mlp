#include <iostream>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"

#include "profiling/Profiler.hpp"

int main() {
    mlp::CUDAProfiler cuda_profiler;

    mlp::Dataset training("res/mnist/mnist_train.csv");
    mlp::DeviceContext context(&cuda_profiler);

    std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };

    mlp::MLP mlp(context, 32);
    mlp.init(layer_sizes);

    mlp::Batch batch(32, 784);
    training.parseBatch(batch);
    mlp.forwardPass(batch);

    float* outputs = new float[32 * 10];

    mlp.copyOutputs(outputs);

    for (size_t batch = 0; batch < 32; batch++) {
        std::cout << "Batch " << batch << ":\n";
        std::cout << outputs[0 + batch * 10];

        for (size_t i = 1; i < 10; i++) {
            std::cout << ", " << outputs[i + batch * 10];
        }

        std::cout << "\n\n";
    }

    delete[] outputs;

    cuda_profiler.print();

    return 0;
}