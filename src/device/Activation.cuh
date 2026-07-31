#pragma once

namespace mlp {
    struct NoActivation {
        __device__ static float activate(float x) {
            return x;
        }
    };

    struct Sigmoid {
        __device__ static float activate(float x) {
            return 1.0f / (1.0f + expf(-x));
        }
    };

    struct Tanh {
        __device__ static float activate(float x) {
            return tanhf(x);
        }
    };

    struct ReLU {
        __device__ static float activate(float x) {
            return fmaxf(0.0f, x);
        }

        __device__ static float derivative(float x) {
            return x >= 0.0f ? 1.0f : 0.0f;
        }
    };

    struct LeakyReLU {
        __device__ static float activate(float x) {
            return x >= 0.0f ? x : 0.01f * x;
        }

        __device__ static float derivative(float x) {
            return x >= 0.0f ? 1.0f : 0.01f;
        }
    };
}