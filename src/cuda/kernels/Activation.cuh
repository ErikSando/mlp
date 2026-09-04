#pragma once

#include <cuda_runtime.h>

namespace mlp {
    namespace cuda {
        constexpr float NO_DERIVATIVE = 1000000.0f;

        struct NoActivation {
            __device__ static float activate(float z) {
                return z;
            }

            __device__ static float derivative(float z) {
                return 1;
            }

            __device__ static float derivative_from_a(float a) {
                return 1;
            }
        };

        struct Sigmoid {
            __device__ static float activate(float z) {
                return 1.0f / (1.0f + expf(-z));
            }

            __device__ static float derivative(float z) {
                float a = activate(z);
                return a * (1 - a);
            }

            __device__ static float derivative_from_a(float a) {
                return a * (1 - a);
            }
        };

        struct Tanh {
            __device__ static float activate(float z) {
                return tanhf(z);
            }

            __device__ static float derivative(float z) {
                float a = activate(z);
                return 1 - a * a;
            }

            __device__ static float derivative_from_a(float a) {
                return 1 - a * a;
            }
        };

        struct ReLU {
            __device__ static float activate(float z) {
                return fmaxf(0.0f, z);
            }

            __device__ static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.0f;
            }

            __device__ static float derivative_from_a(float a) {
                return a >= 0.0f ? 1.0f : 0.0f;
            }
        };

        struct LeakyReLU {
            __device__ static float activate(float z) {
                return z >= 0.0f ? z : 0.01f * z;
            }

            __device__ static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.01f;
            }

            __device__ static float derivative_from_a(float a) {
                return a >= 0.0f ? 1.0f : 0.01f;
            }
        };

        struct Softmax {
            __device__ static float derivative(float z) {
                return NO_DERIVATIVE;
            }

            __device__ static float derivative_from_a(float a) {
                return a * (1 - a);
            }
        };
    }
}