#include <iostream>

#include "data/Dataset.hpp"
#include "nn/InputNodes.hpp"

int main() {
    mlp::Dataset training("res/mnist/mnist_train.csv");

    std::vector<mlp::InputNodes> inputs;

    training.parseBatch(3, 4, inputs);

    return 0;
}