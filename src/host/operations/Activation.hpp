#pragma once

#include <cmath>

namespace mlp {
    namespace host {
        constexpr float NO_DERIVATIVE = 1000000.0f;

        struct NoActivation {
            static float activate(float z) {
                return z;
            }

            static float derivative(float z) {
                return 1;
            }

            static float derivative_from_a(float a) {
                return 1;
            }
        };

        struct Sigmoid {
            static float activate(float z) {
                return 1.0f / (1.0f + expf(-z));
            }

            static float derivative(float z) {
                float a = activate(z);
                return a * (1 - a);
            }

            static float derivative_from_a(float a) {
                return a * (1 - a);
            }
        };

        struct Tanh {
            static float activate(float z) {
                return tanhf(z);
            }

            static float derivative(float z) {
                float a = activate(z);
                return 1 - a * a;
            }

            static float derivative_from_a(float a) {
                return 1 - a * a;
            }
        };

        struct ReLU {
            static float activate(float z) {
                return z >= 0.0f ? z : 0.0f; // i think this is faster than using fmaxf
                // return fmaxf(0.0f, z);
            }

            static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.0f;
            }

            static float derivative_from_a(float a) {
                return a >= 0.0f ? 1.0f : 0.0f;
            }
        };

        struct LeakyReLU {
            static float activate(float z) {
                return z >= 0.0f ? z : 0.01f * z;
            }

            static float derivative(float z) {
                return z >= 0.0f ? 1.0f : 0.01f;
            }

            static float derivative_from_a(float a) {
                return a >= 0.0f ? 1.0f : 0.01f;
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