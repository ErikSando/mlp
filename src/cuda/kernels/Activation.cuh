#pragma once

namespace mlp {
    namespace cuda {
        struct NoActivation {
            __device__ static float activate(float z) {
                return z;
            }
        };

        struct Sigmoid {
            __device__ static float activate(float z) {
                return 1.0f / (1.0f + expf(-z));
            }
        };

        struct Tanh {
            __device__ static float activate(float z) {
                return tanhf(z);
            }
        };

        struct ReLU {
            __device__ static float activate(float z) {
                return fmaxf(0.0f, z);
            }

            __device__ static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.0f;
            }
        };

        struct LeakyReLU {
            __device__ static float activate(float z) {
                return z >= 0.0f ? z : 0.01f * z;
            }

            __device__ static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.01f;
            }
        };
    }
}