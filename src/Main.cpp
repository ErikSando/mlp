#include <iostream>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"

int main() {
    mlp::Dataset training("res/mnist/mnist_train.csv");

    mlp::DeviceContext context;

    std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };

    mlp::MLP mlp(context, layer_sizes, 32);

    mlp::Batch batch(32, 784);

    training.parseBatch(batch);

    std::cout << batch.data.size() << "\n";

    size_t index = 0;

    for (size_t i = 0; i < batch.size; i++) {
        std::cout << "Label: " << batch.labels[i] << "\n";

        std::cout << batch.data[index];

        for (size_t j = index + 1; j < index + 784; j++) {
            std::cout << "," << batch.data[j];
        }

        std::cout << "\n\n";

        index += 784;
    }

    return 0;
}