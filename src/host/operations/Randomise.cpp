#include <random>

#include "host/Context.hpp"

namespace mlp {
    namespace host {
        void Context::randomise(Matrix& matrix, float min, float max) const {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(min, max);

            float* data = matrix.data();

            for (int i = 0; i < matrix.size(); i++) {
                data[i] = dist(gen);
            }
        }
    }
}