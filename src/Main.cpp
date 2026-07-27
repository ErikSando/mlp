#include <iostream>

#include <random>

#include "device/DeviceContext.hpp"

#include "profiling/Profiler.hpp"

int main() {
    mlp::CUDAProfiler cuda_profiler;
    mlp::Profiler profiler;

    mlp::DeviceContext context(&cuda_profiler);

    std::vector<float> A(256 * 256);
    std::vector<float> B(256 * 256);
    std::vector<float> C(256 * 256);
    std::vector<float> D(256 * 256);

    mlp::Matrix mA(256, 256);
    mlp::Matrix mB(256, 256);
    mlp::Matrix mC(256, 256);
    mlp::Matrix mD(256, 256);

    for (int i = 0; i < A.size(); i++) {
        A[i] = std::rand() * 20 - 10;
        B[i] = std::rand() * 20 - 10;
    }

    cuda_profiler.disable();

    context.transfer(A.data(), mA);
    context.transfer(B.data(), mB);

    cuda_profiler.enable();

    for (int i = 0; i < 5; i++) {
        context.multiplyOld(mA, mB, mC);
        context.multiply(mA, mB, mD);
    }

    cuda_profiler.disable();
    cuda_profiler.print();

    context.transfer(mC, C.data());
    context.transfer(mD, D.data());

    for (int i = 0; i < C.size(); i++) {
        if (C[i] != D[i]) {
            std::cout << "Incorrect result\n";
            break;
        }
    }

    return 0;
}