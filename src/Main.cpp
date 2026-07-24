#include <iostream>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"

int main() {
    mlp::Dataset training("res/mnist/mnist_train.csv");

    mlp::DeviceContext context;

    std::vector<size_t> layer_sizes = { 10, 10, 10, 10 };

    mlp::MLP mlp(context, layer_sizes);

    return 0;
}