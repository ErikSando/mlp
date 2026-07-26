#include <iostream>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"

void printMatrix(float* matrix, int size) {
    std::cout << matrix[0];

    for (int i = 1; i < size; i++) {
        std::cout << "   " << matrix[i];
    }

    std::cout << "\n";
}

int main() {
    // mlp::Dataset training("res/mnist/mnist_train.csv");
    mlp::DeviceContext context;

    // std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };

    // mlp::MLP mlp(context, layer_sizes, 32);

    // mlp::Batch batch(32, 784);
    // training.parseBatch(batch);
    // mlp.forwardPass(batch);

    // float* outputs = new float[32 * 10];

    // mlp.copyOutputs(outputs);

    // for (size_t batch = 0; batch < 32; batch++) {
    //     std::cout << "Batch " << batch << ":\n";
    //     std::cout << outputs[0 + batch * 10];

    //     for (size_t i = 1; i < 10; i++) {
    //         std::cout << ", " << outputs[i + batch * 10];
    //     }

    //     std::cout << "\n\n";
    // }

    // delete[] outputs;

    float M[9] = {
        -10, 10,  3,
         -1,  1, -3,
         -5,  5, 15
    };

    mlp::Matrix input(3, 3);
    mlp::Matrix output(3, 3);

    context.transfer(M, input);

    context.sigmoid(input, output);
    context.transfer(output, M);

    printMatrix(M, 9);
    std::cout << "\n";

    context.tanh(input, output);
    context.transfer(output, M);

    printMatrix(M, 9);
    std::cout << "\n";

    context.relu(input, output);
    context.transfer(output, M);

    printMatrix(M, 9);
    std::cout << "\n";

    context.leakyReLU(input, output);
    context.transfer(output, M);

    printMatrix(M, 9);
    std::cout << "\n";

    return 0;
}