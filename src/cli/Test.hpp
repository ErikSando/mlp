#pragma once

#include <cstddef>

namespace mlp {
    struct TestData {
        void reset() {
            incorrect = 0UL;
            correct = 0UL;
        }

        float getAccuracy() { return (float) correct / (float) (correct + incorrect); }

        size_t incorrect;
        size_t correct;
    };
}