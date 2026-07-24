#include <iostream>

#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
#include "mlp/MLP.hpp"

int main() {
    mlp::Dataset training("res/mnist/mnist_train.csv");
    mlp::DeviceContext context;

    float A[9] = {
        2, 3, 1,
        1, 2, 3,
        3, 1, 2
    };

    float B[9] = {
        2, 1, 3,
        3, 2, 1,
        1, 3, 2
    };

    float C[9], D[9];

    mlp::Matrix mA(3, 3), mB(3, 3), mC(3, 3), mD(3, 3);

    context.transfer(A, mA);
    context.transfer(B, mB);

    context.add(mA, mB, mC);
    context.multiply(mA, mB, mD);

    context.transfer(mC, C);
    context.transfer(mD, D);

    for (int i = 0; i < 8; i++) {
        if (i % 3 == 0) std::cout << "\n";
        std::cout << " " << A[i] << ",";
    }
    std::cout << " " << A[8] << "\n";

    for (int i = 0; i < 8; i++) {
        if (i % 3 == 0) std::cout << "\n";
        std::cout << " " << B[i] << ",";
    }
    std::cout << " " << B[8] << "\n\n";

    std::cout << "Addition:\n";
    for (int i = 0; i < 8; i++) {
        if (i % 3 == 0) std::cout << "\n";
        std::cout << " " << C[i] << ",";
    }
    std::cout << " " << C[8] << "\n\n";

    std::cout << "Multiplication:\n";
    for (int i = 0; i < 8; i++) {
        if (i % 3 == 0) std::cout << "\n";
        std::cout << " " << D[i] << ",";
    }
    std::cout << " " << D[8] << "\n\n";

    std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };

    mlp::MLP mlp(context, layer_sizes, 32);

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

    return 0;
}