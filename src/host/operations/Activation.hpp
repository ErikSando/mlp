#pragma once

#include <cmath>

namespace mlp {
    namespace host {
        constexpr float NO_DERIVATIVE = 1000000.0f;

        struct NoActivation {
            static float activate(float z) {
                return z;
            }
        };

        struct Sigmoid {
            static float activate(float z) {
                return 1.0f / (1.0f + expf(-z));
            }
        };

        struct Tanh {
            static float activate(float z) {
                return tanhf(z);
            }
        };

        struct ReLU {
            static float activate(float z) {
                return fmaxf(0.0f, z);
            }

            static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.0f;
            }
        };

        struct LeakyReLU {
            static float activate(float z) {
                return z >= 0.0f ? z : 0.01f * z;
            }

            static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.01f;
            }
        };

        struct Softmax {
            static float derivative(float z) {
                return NO_DERIVATIVE;
            }

            static float derivative_from_a(float a) {
                return a * (1 - a);
            }
        };
    }
}